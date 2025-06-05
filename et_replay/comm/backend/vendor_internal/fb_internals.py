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


logger = logging.getLogger(__name__)
logger.info("importing FB internals.")

try:
    # import pytorch_mtia_backend to register mtia backend to use mtia/hccl
    import param_bench.et_replay.comm.backend.vendor_internal.pytorch_mtia_backend  # noqa
except ImportError as e:
    print(f"ImportError: {e}")
    logger.info(
        "pytorch_mtia_backend not found. Build PARAM Bench with '-c fbcode.use_hccl=True' to enable it."
    )
    pass
