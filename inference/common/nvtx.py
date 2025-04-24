# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import contextlib
from functools import wraps

import torch


@contextlib.contextmanager
def add_nvtx_event(event_name):
    """
    Context manager to add an NVTX event around a code block.

    Args:
    - event_name: The name of the event to be recorded.
    """
    torch.cuda.nvtx.range_push(event_name)
    yield
    torch.cuda.nvtx.range_pop()


def instrument_nvtx(func):
    """
    Decorator that records an NVTX range for the duration of the function call.

    Args:
    - func: The function to be decorated.

    Returns:
    - Wrapped function that is now being profiled.
    """

    @wraps(func)
    def wrapped_fn(*args, **kwargs):
        with add_nvtx_event(func.__qualname__):
            ret_val = func(*args, **kwargs)
        return ret_val

    return wrapped_fn
