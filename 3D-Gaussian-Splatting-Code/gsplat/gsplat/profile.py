import os
import time
from functools import wraps
from typing import Callable, Optional

import torch

profiler = {}


class timeit(object):
    """Profiler that is controled by the TIMEIT environment variable.

    If TIMEIT is set to 1, the profiler will measure the time taken by the decorated function.

    Usage:

    ```python
    @timeit()
    def my_function():
        pass

    # Or

    with timeit(name="stage1"):
        my_function()

    print(profiler)
    ```
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.start_time: Optional[float] = None
        self.enabled = os.environ.get("TIMEIT", "0") == "1"

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time = end_time - self.start_time
            if self.name not in profiler:
                profiler[self.name] = total_time
            else:
                profiler[self.name] += total_time

    def __call__(self, f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            with self:
                self.name = f.__name__
                return f(*args, **kwargs)

        return decorated
