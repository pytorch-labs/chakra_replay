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

import datetime
import gzip
import json
import logging
import os
import uuid
from typing import Any

from et_replay.execution_trace import ExecutionTrace


def get_tmp_trace_filename() -> str:
    """Generate a temporary filename using the current date, a UUID, and the process ID."""
    trace_fn = (
        "tmp_"
        + datetime.datetime.today().strftime("%Y%m%d")
        + "_"
        + uuid.uuid4().hex[:7]
        + "_"
        + str(os.getpid())
        + ".json"
    )
    return trace_fn


def trace_handler(prof: Any) -> None:
    """Export a chrome trace"""
    fn = get_tmp_trace_filename()
    prof.export_chrome_trace("/tmp/" + fn)
    logging.warning(f"Chrome profile trace written to /tmp/{fn}")


def load_execution_trace_file(et_file_path: str) -> ExecutionTrace:
    """Loads Execution Trace from json file and parses it."""
    data = read_dictionary_from_json_file(et_file_path)
    return ExecutionTrace(data)


def read_dictionary_from_json_file(file_path: str) -> dict[Any, Any]:
    """Read a json file and return it as a dictionary."""
    with (
        gzip.open(file_path, "rb") if file_path.endswith("gz") else open(file_path)
    ) as f:
        return json.load(f)


def write_dictionary_to_json_file(file_path: str, data: dict[Any, Any]) -> None:
    """Write input dictionary to a json file."""
    if file_path.endswith("gz"):
        with gzip.open(file_path, "w") as f:
            f.write(json.dumps(data, indent=4).encode("utf-8"))
    else:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
