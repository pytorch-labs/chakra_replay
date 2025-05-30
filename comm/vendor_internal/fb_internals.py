import asyncio
import gzip
import json
import logging
import random
import re
from io import BytesIO, StringIO

import torch
from et_replay.comm.param_profile import paramProfile

from manifold.clients.python import ManifoldClient

logger = logging.getLogger(__name__)
logger.info("importing comm FB internals.")

import caffe2.torch.fb.hpc.extend_distributed as extend_distributed  # noqa


def all_to_allv_internal(collectiveArgs):
    assert collectiveArgs.all2all_qcomm is not None
    assert collectiveArgs.ipTensor.dtype == torch.float32
    output_split_sizes = collectiveArgs.opTensor_split
    input_split_sizes = collectiveArgs.ipTensor_split

    output_split_sizes = output_split_sizes if output_split_sizes is not None else []
    input_split_sizes = input_split_sizes if input_split_sizes is not None else []

    # quantization before alltoall
    _inSize = (
        collectiveArgs.ipTensor.nelement() * collectiveArgs.ipTensor.element_size()
    )
    _outSize = (
        collectiveArgs.opTensor.nelement() * collectiveArgs.opTensor.element_size()
    )
    _sizes = f"(send: {_inSize}, recv: {_outSize} bytes; width {collectiveArgs.all2all_qcomm.fwd_comm_precision}) #"
    with paramProfile(
        timer=collectiveArgs.quant_time,
        description=f"# PARAM: Alltoall quantization {_sizes} #",
    ):
        if collectiveArgs.use_ext_dist:
            dist_group = collectiveArgs.group.my_pg
        else:
            dist_group = collectiveArgs.group
        work = collectiveArgs.all2all_qcomm.gen_alltoall_work(
            collectiveArgs.ipTensor,
            collectiveArgs.opTensor,
            input_split_sizes,
            output_split_sizes,
            dist_group,
        )
        work.prepare_quantized_tensors()
        if not collectiveArgs.asyncOp:
            torch.cuda.synchronize(collectiveArgs.device)

    # blocking alltoall
    with paramProfile(description=f"# PARAM: quantized Alltoall {_sizes} #"):
        work.do_all2all_work()
        if not collectiveArgs.asyncOp:
            torch.cuda.synchronize(collectiveArgs.device)

    # de-quantization after alltoall
    with paramProfile(
        timer=collectiveArgs.dequant_time,
        description=f"# PARAM: Alltoall de-quantization {_sizes} #",
    ):
        work.dequantize_tensor()
        if not collectiveArgs.asyncOp:
            torch.cuda.synchronize(collectiveArgs.device)


def all_to_all_internal(collectiveArgs):
    return all_to_allv_internal(collectiveArgs)


async def manifoldRead(remotePath, bucket="param"):
    logger.info(f"open bucket {bucket} and read {remotePath}")
    with ManifoldClient.get_client(bucket) as client:
        # try add .gz if .json is not found
        if not client.sync_exists(remotePath) and remotePath.endswith(".json"):
            remotePath += ".gz"
        for i in range(5):
            try:
                stream = BytesIO()
                await client.get(remotePath, stream)  # read into stream
                trace = stream.getvalue()
            except Exception as e:
                logger.warning(f"Manifold client attempt {i} had the error: {e}")
                await asyncio.sleep(random.randrange(1, 5))
            else:
                break
        else:
            raise Exception(
                "Could not download from Manifold. See previous errors. Enable --log WARNING if needed."
            )

    if remotePath.endswith(".gz"):
        traceList = StringIO(gzip.decompress(trace).decode("utf-8"))
    else:
        traceList = StringIO(trace.decode("utf-8"))

    return traceList


async def manifoldWrite(
    output, remotePath, bucket="param", enable_compression=True, expire_time=None
):
    logger.debug(f"open bucket {bucket} and read {remotePath}")
    with ManifoldClient.get_client(bucket) as client:
        prefix_path = remotePath.rsplit("/", 1)[0]
        try:
            await client.mkdir(prefix_path, recursive=True)
        except Exception:
            # ignore exception as the directory may already exist
            pass
        if enable_compression:
            stream = gzip.compress(json.dumps(output, indent=2).encode())
            remotePath += ".gz"
        else:
            stream = BytesIO(json.dumps(output, indent=2).encode())
        await client.put(
            remotePath,
            stream,
            predicate=ManifoldClient.Predicates.AllowOverwrite,
            ttl=expire_time,
        )  # write from stream


def manifoldParse(remotePath):
    # only support manifold now (bunnylol "manifold")
    # acceptable format: "manidfold://<bucket>/path/to/trace"
    remotePathSplit = remotePath.split("://", 1)
    protocol = remotePathSplit[0]
    if protocol not in ("manifold", "Manifold"):
        logger.warning(
            f"Currently only support manifold for FB internal use, specified protocol '{protocol}' may not be supported (typo?)"
        )
    manifoldUrlSplit = remotePathSplit[1].split("/", 1)
    return (manifoldUrlSplit[0], manifoldUrlSplit[1])


def writeRemoteTrace(
    output,
    remotePath="manifold://param/tree/comms_traces/ads15x-dpp-np128_replayed/rank0.json",
    enable_compression=True,
):
    (bucketName, bucketPath) = manifoldParse(remotePath)
    return asyncio.run(
        manifoldWrite(
            output=output,
            bucket=bucketName,
            remotePath=bucketPath,
            enable_compression=enable_compression,
        )
    )


def getExecutionTracePath(remotePath, rank):
    """
    Execution Trace(ET) name have a timestamp which is different for each rank
    ET input has two scenrios:
    Case 1: all ranks' ET trace name are combined into one RemotePath
    Case 2: in pytorch_execution_trace bucket, traces under a folder end with /{mast_job_name}/{attempt}
    Case 3: trace folder name is provided and needs to get trace name by each rank itself
    TODO: Case 3 is basically for existing traces and generated traces. May need to be deprecated later.
    """
    etPath = ""
    # Case 1: e.g. remotePath = "<MANIFOLD PATH for RANK-0>,<MANIFOLD PATH for RANK-1>,..."
    if "," in remotePath:
        for path in remotePath.split(","):
            if f"rank-{rank}." in path:
                logger.info(f"[rank-{rank}] Found remote path {path}")
                return path
        raise Exception(f"[rank-{rank}] No trace found in remotePath {remotePath}")
    else:  # Case 2 and Case 3
        (bucketName, bucketPath) = manifoldParse(remotePath)
        if bucketName not in ("pytorch_execution_trace", "pyper_traces", "param"):
            # e.g. bucketPath "tree/traces/dynocli/sw-973922241-OfflineTraining-w3hfd5st27g2xc/rank-0"
            bucketPath += f"/rank-{rank}"
            remotePath += f"/rank-{rank}"
        # TODO: use scuba query for obtain relative ET traces
        filename = asyncio.run(
            getTraceFilenameFromManifold(
                bucketPath,
                bucketName,
                rank,
            )
        )
        if not filename:
            raise Exception(f"No trace found for rank-{rank} at {remotePath}")
        else:
            etPath = f"{remotePath}/{filename}"
    return etPath


def readRemoteTrace(
    remotePath="manifold://param/tree/comms_traces/ads15x-dpp-np128",
    rank=0,
    full_trace_path=False,
    trace_type="basic",
):
    if trace_type == "et":
        remotePath = getExecutionTracePath(remotePath, rank)
        full_trace_path = True  # ET trace path is always full path
    (bucketName, bucketPath) = manifoldParse(remotePath)
    if not full_trace_path:
        bucketPath = f"{bucketPath}/rank{rank}.json"
    return asyncio.run(manifoldRead(bucket=bucketName, remotePath=bucketPath))


_torch_profiler = None


async def getTraceFilenameFromManifold(path: str, bucket: str, rank: int) -> str:
    logger.info(
        f"Getting trace manifold filename by listing bucket: {bucket} path: {path}"
    )
    with ManifoldClient.get_client(bucket) as client:
        try:
            async for filename, _ in client.ls(path=path):
                # As defined, et_manifold_file_name = trace_file_prefix + ".et.json.gz"
                match = re.search(
                    rf"rank-{str(rank)}\.(.*)",
                    filename,
                )
                if match:
                    logger.info(f"Found ET trace file: {filename}")
                    return filename
        except Exception as e:
            logger.warning(f"Manifold client list had the error: {e}")
    return ""
