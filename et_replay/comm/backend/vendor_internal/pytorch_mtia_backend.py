# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import mtia.host_runtime.mtia_streaming.python.fbia_streaming_bindings as fbia_streaming

import torch
import torch.distributed as dist
from et_replay.comm.backend.base_backend import register_customized_backend

from et_replay.comm.backend.pytorch_dist_backend import PyTorchDistBackend

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PyTorchMtiaBackend(PyTorchDistBackend):
    def get_collective_group(self, collectiveArgs):
        return collectiveArgs.group

    def device_sync(self, collectiveArgs):
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        if dev_str == "mtia":
            fbia_streaming.sync_stream(self.fbia_stream)
        else:
            logger.warning(f"device type {dev_str} not supported for HCCL backend")

    def barrier(self, collectiveArgs, name="dummy", retFlag=False):
        my_dev = self.get_device()
        retObj = dist.barrier(
            collectiveArgs.group,
            async_op=collectiveArgs.asyncOp,
            # FIXME: device_ids only support nccl now, need support from distributed_c10d before hccl can use it
            device_ids=(
                [my_dev.index]
                if "nccl" in dist.get_backend(collectiveArgs.group)
                else None
            ),
        )

        if collectiveArgs.asyncOp:
            collectiveArgs.waitObj.append(retObj)

        if retFlag:
            return retObj

    def barrier_all_ranks(self):
        dist.barrier(
            device_ids=[self.get_device().index]
            if dist.get_backend() == "hccl"
            else None
        )

    # Compute functions

    def alloc_random(
        self, sizeArr, curRankDevice="mtia", dtype=torch.float32, scaleFactor=1.0
    ):
        curRankDevice_orig = curRankDevice
        if curRankDevice_orig == "mtia":
            curRankDevice = "cpu"

        if dtype in (torch.uint8, torch.int16, torch.int32, torch.long):
            ipTensor = torch.randint(
                low=0, high=10, size=sizeArr, device=curRankDevice, dtype=dtype
            )
        elif dtype == torch.bool:
            ipTensor = (
                torch.rand(sizeArr, device=curRankDevice, dtype=torch.float32) < 0.5
            )
        else:
            ipTensor = torch.rand(sizeArr, device=curRankDevice, dtype=dtype)
            if (scaleFactor) != 0:
                ipTensor = ipTensor / scaleFactor

        if curRankDevice_orig == "mtia":
            ipTensor = ipTensor.to(device=curRankDevice_orig)
            curRankDevice = curRankDevice_orig

        return ipTensor

    def clear_memory(self, collectiveArgs):
        if collectiveArgs.ipTensor is not None:
            del collectiveArgs.ipTensor

        if collectiveArgs.opTensor is not None:
            del collectiveArgs.opTensor

        if collectiveArgs.ipTensor_pair is not None:
            del collectiveArgs.ipTensor_pair
            del collectiveArgs.opTensor_pair

    def get_device(self):
        """get current device: 'cpu' or 'mtia'"""
        # TODO: this is a temporary workaround; need to unify the type of commsParams in comms and dlrm
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        my_dev = torch.device(dev_str)
        if dev_str == "mtia":
            # explicitly select the device ordinal based on the local rank
            ordinal = self.get_local_rank()
            if self.get_local_rank() == -1:
                logger.warning(
                    "Cannot determine device ordinal since LOCAL_RANK is -1. Try Device 0 and continue. "
                )
                ordinal = 0
            my_dev = torch.device(f"mtia:{ordinal}")

        return my_dev

    def set_device(self, local_rank, global_rank):
        """set current device: 'cpu' or 'mtia'"""
        dev_str = (
            self.commsParams["device"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.device
        )
        if dev_str.startswith("mtia"):
            self.fbia_device = fbia_streaming.current_ordinal()
            self.fbia_stream = fbia_streaming.current_stream()

        logger.info(
            f"rank {global_rank} set torch device to mtia:{self.fbia_device} on stream {self.fbia_stream}"
        )

    def get_new_stream(self):
        """get/allocate a new stream"""
        if self.commsParams.device == "mtia":
            current_ordinal: int = fbia_streaming.current_ordinal()
            return fbia_streaming.create_stream(current_ordinal)
        else:
            return None

    def switch_stream(self, stream, device: torch.device | None):
        """switch to a new stream and return the current stream"""
        if device is None:
            device = self.get_device()
        if stream is not None and device.type == "mtia":
            cur_stream = fbia_streaming.current_stream()
            fbia_streaming.set_current_stream(stream)
            return cur_stream
        else:
            return None

    # Init functions
    def __init__(self, bootstrap_info, commsParams):
        super().__init__(bootstrap_info, commsParams)

        backend = (
            self.commsParams["backend"]
            if isinstance(self.commsParams, dict)
            else self.commsParams.backend
        )

        # Import hccl pg plugin if used
        if backend == "hccl":
            try:
                import hccl.pg  # noqa
                import mtia.host_runtime.torch_mtia.dynamic_library  # noqa
            except ImportError as e:
                raise RuntimeError(
                    f"Unable to import HCCL PG Plugin or mtia backend: {e}"
                )
            else:
                torch.mtia.init()
                fbia_streaming.init()

        self.fbia_stream = fbia_streaming.CURRENT_STREAM_ID
        self.fbia_device = None

    def initialize_backend(
        self, master_ip, master_port, backend="hccl", eager_mode=None
    ):
        self.set_device(self.bootstrap_info.local_rank, self.bootstrap_info.global_rank)

        global_rank = self.bootstrap_info.global_rank
        world_size = self.bootstrap_info.world_size

        self.use_ext_dist = False

        if self.tcp_store is None:
            # TCP store initializaiton for generic CPU data
            self.tcp_store = dist.TCPStore(
                master_ip,
                int(master_port),
                world_size,
                is_master=(global_rank == 0),
                use_libuv=True,
            )

        if not dist.is_initialized():
            # init default process group if not yet initialized or extend_distributed failed or is disabled
            dist.init_process_group(
                backend="cpu:gloo,mtia:hccl",
                rank=global_rank,
                world_size=world_size,
                store=self.tcp_store if self.commsParams.init_method is None else None,
                init_method=self.commsParams.init_method,
            )

        # default 1 group, maybe overwritten by user created groups via initialize_groups
        self.groups = {}
        self.groups[0] = self.get_default_group()
        self.num_pgs = len(self.groups)


logger.info("Registering HCCL backend for MTIA device")
register_customized_backend("hccl", PyTorchMtiaBackend, "mtia")
