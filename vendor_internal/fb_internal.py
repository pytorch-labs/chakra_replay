import gzip
import logging
import logging as logger
import os
from io import BytesIO, StringIO

from aiplatform.monitoring.unitrace.upload_manifold import export_trace_func

from manifold.clients.python import ManifoldClient
from torch.profiler import ProfilerActivity


def add_internal_skip_nodes(skip_nodes):
    internal_skip_nodes = ["fb::", "pyspeech::"]
    return skip_nodes + internal_skip_nodes


def add_internal_parallel_nodes_parents(parallel_nodes_parents):
    internal_parallel_nodes_parents = [
        "## sparse_data_dist ##",
        "## wait_sparse_data_dist ##",
        "## load_batch ##",
    ]
    return parallel_nodes_parents + internal_parallel_nodes_parents


def add_internal_label():
    internal_label = "ProfilerStep#"
    return internal_label


def read_remote_skip_node_file(remote_path):
    try:
        remote_path_split = remote_path.split("://", 1)
        protocol = remote_path_split[0]
        if protocol not in ("manifold", "Manifold"):
            logging.warning(
                f"Currently only support manifold for FB internal use, specified protocol '{protocol}' may not be supported (typo?)"
            )
        manifoldUrlSplit = remote_path_split[1].split("/", 1)
        bucket = manifoldUrlSplit[0]
        path = manifoldUrlSplit[1]
        with ManifoldClient.get_client(bucket) as client:
            file_name = None
            for e in client.sync_ls(path):
                if e[0] == "skip-node.json":
                    file_name = f"{path}/{e[0]}"
                    break
            if file_name is not None:
                logging.info(
                    f"Read skip node file at {protocol}://{bucket}/{file_name}"
                )
                stream = BytesIO()
                client.sync_get(file_name, stream)
                file_stream = stream.getvalue()
                json_string = StringIO(file_stream.decode("utf-8"))
                return json_string
            else:
                return None
    except Exception as e:
        logging.error("Cannot read remote skip-node.json, error: ", e)
        exit(1)


def read_remote_trace(remote_path):
    try:
        remote_path_split = remote_path.split("://", 1)
        protocol = remote_path_split[0]
        if protocol not in ("manifold", "Manifold"):
            logging.warning(
                f"Currently only support manifold for FB internal use, specified protocol '{protocol}' may not be supported (typo?)"
            )
        manifoldUrlSplit = remote_path_split[1].split("/", 1)
        bucket = manifoldUrlSplit[0]
        path = manifoldUrlSplit[1]
        with ManifoldClient.get_client(bucket) as client:
            # If input is the exact path to the et.
            if path.endswith(".json") or path.endswith(".gz"):
                trace_path = path
                # Some et files contains time stamp, need to handle it.
                dir_name = os.path.dirname(path)
                # file_root_name is like rank-0.json
                file_root_name = path.split("/", -1)[-1].rstrip(".json")
                for e in client.sync_ls(dir_name):
                    if file_root_name in e[0]:
                        trace_path = f"{dir_name}/{e[0]}"
                        break
            # If input is a directory that contains the et.
            else:
                for e in client.sync_ls(path):
                    if "et" in e[0]:
                        trace_path = f"{path}/{e[0]}"
                        break
            logging.info(f"Read trace at {protocol}://{bucket}/{trace_path}")
            stream = BytesIO()
            client.sync_get(trace_path, stream)
            et_stream = stream.getvalue()
            if trace_path.endswith(".gz"):
                et = StringIO(gzip.decompress(et_stream).decode("utf-8"))
            else:
                et = StringIO(et_stream.decode("utf-8"))
            return et, f"{protocol}://{bucket}/{trace_path}"

    except Exception as e:
        logging.error("Cannot access remote trace, error: ", e)
        exit(1)


def generate_query_url(start_time, end_time, cuda_id):
    import datetime
    import os
    import socket

    import torch
    from libfb.py.scuba_url import ScubaDrillstate, ScubaURL

    drillstate = ScubaDrillstate()
    unix_start_time = datetime.datetime.timestamp(start_time) * 1000
    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    unix_end_time = datetime.datetime.timestamp(end_time) * 1000
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    drillstate.setStartTime(start_time)
    drillstate.setEndTime(end_time)
    host_name = socket.gethostname()
    drillstate.addConstraint(
        "host_name", "eq", host_name[: host_name.find(".", host_name.find(".") + 1)]
    )
    if cuda_id == -1:
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            drillstate.addConstraint("device", "eq", os.getenv("CUDA_VISIBLE_DEVICES"))
        else:
            drillstate.addConstraint("device", "eq", torch.cuda.current_device())
    else:
        drillstate.addConstraint("device", "eq", cuda_id)
    drillstate.setSampleCols(
        [
            "gpu_device_utilization",
            "gpu_memory_utilization",
            "gpu_power_draw",
            "gflops_per_sec_estimated",
            "time",
            "hbm_mem_bw_gbps",
            "hbm_mem_bw_util",
            "system_mem_bw_mbps",
            "system_cpu_s_util",
            "sm_utilization",
            "power_usage_total",
        ]
    )
    dataset = "gpu_dyno_stats"
    url = ScubaURL(dataset=dataset, drillstate=drillstate)
    logging.warning(
        f"Start time: {start_time} unix {unix_start_time}, end time: {end_time}, unix {unix_end_time}"
    )
    logging.warning(f"Scuba query url is: {url.fburl()}")


def initialize_collectiveArgs_internal(collectiveArgs, commsParams):
    from caffe2.torch.fb.hpc import quantized_comms_lib as qcomms_lib

    bv2CommType = {
        2: qcomms_lib.CommType.INT2,
        4: qcomms_lib.CommType.INT4,
        8: qcomms_lib.CommType.INT8,
        16: qcomms_lib.CommType.FP16,
        32: qcomms_lib.CommType.FP32,
    }

    collectiveArgs.all2all_qcomm = qcomms_lib.QuantizedAll2AllContext(
        bv2CommType[commsParams.bitwidth],
        bv2CommType[commsParams.bitwidth],
        commsParams.quant_a2a_embedding_dim,
        delay_quant=True,  # delay the quantization to measure overhead separately
    )

    # not implemented currently in comms.py, but ok.
    try:
        collectiveArgs.reducescatter_allgather_qcomm = (
            qcomms_lib.QuantizedReduceScatterContext(
                bv2CommType[commsParams.bitwidth], bv2CommType[commsParams.bitwidth]
            )
        )
    except NotImplementedError:
        logger.warning(
            f"cannot support quantization with bitwidth {commsParams.bitwidth} for Reduce-Scatter/Allgather"
        )

    collectiveArgs.allreduce_qcomm = commsParams.bitwidth
    collectiveArgs.reduce_qcomm = commsParams.bitwidth


def remove_quantization_handlers(collectiveArgs):
    collectiveArgs.all2all_qcomm = None
    collectiveArgs.reducescatter_allgather_qcomm = None
    collectiveArgs.allreduce_qcomm = 32
    collectiveArgs.reduce_qcomm = 32


def get_fb_profiler_activities(device):
    activities = {}
    if device == "cpu":
        activities = {ProfilerActivity.CPU}
    elif device == "cuda":
        activities = {ProfilerActivity.CPU, ProfilerActivity.CUDA}
    elif device == "mtia":
        activities = {ProfilerActivity.CPU, ProfilerActivity.MTIA}

    return activities


def get_fb_profiler_trace_handler(rank):
    return export_trace_func(
        "/tmp",
        worker_name=f"rank-{rank}",
        bucket_name="hpc_traces",
        zoomer_request_callsite="hpc",
    )
