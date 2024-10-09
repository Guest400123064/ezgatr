import functools

import torch


def time_cuda_exec(n_exec: int = 1000, report_avg: bool = True):
    r"""Time the execution of a function involving CUDA."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            res = []
            for _ in range(n_exec):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)

                s.record()
                func(*args, **kwargs)
                e.record()

                torch.cuda.synchronize()
                res.append(s.elapsed_time(e))

            if report_avg:
                print(f"Average time: {sum(res) / n_exec} ms")

        return _wrapper

    return _decorator
