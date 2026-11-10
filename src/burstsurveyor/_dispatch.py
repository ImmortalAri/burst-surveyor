"""Memory-aware scheduling. Only the coordinator commits completed worker output."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import time

import psutil

MIB = 1024**2
RESERVE_MIB = 512
MAX_WORKERS = 3
MEMORY_WAIT_SECONDS = 30


def estimate_bytes(event):
    return MIB * max(
        384, 192 + 6 * max(event["input_bytes"], event["expanded_bytes"]) / MIB
    )


def configure_worker():
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"


def dispatch(events, workers, make_task, work, commit, *, defer=None):
    """Return stage telemetry. I/O or worker exceptions leave atomic commits resumable."""
    pending = sorted(events, key=estimate_bytes, reverse=True)
    active = {}
    peak = 0
    smallest_available = psutil.virtual_memory().available
    idle_start = None
    pool = None
    try:
        while pending or active:
            free = psutil.virtual_memory().available
            smallest_available = min(smallest_available, free)
            if pending and len(active) < workers:
                index = next(
                    (
                        i
                        for i, event in enumerate(pending)
                        if free >= RESERVE_MIB * MIB + estimate_bytes(event)
                    ),
                    None,
                )
                if index is not None:
                    event = pending.pop(index)
                    task = make_task(event)
                    idle_start = None
                    if workers == 1:
                        peak = 1
                        commit(event, task, work(task))
                    else:
                        if pool is None:
                            pool = ProcessPoolExecutor(
                                max_workers=workers,
                                initializer=configure_worker,
                                mp_context=multiprocessing.get_context("spawn"),
                            )
                        future = pool.submit(work, task)
                        active[future] = (event, task)
                        if len(active) > peak:
                            peak = len(active)
                            time.sleep(0.25)
                    continue
            if active:
                done, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    event, task = active.pop(future)
                    commit(event, task, future.result())
                continue
            if pool is not None:
                # Idle imported workers can prevent even the smallest remaining event from fitting.
                pool.shutdown(wait=True)
                pool = None
                continue
            idle_start = idle_start if idle_start is not None else time.monotonic()
            if time.monotonic() - idle_start >= MEMORY_WAIT_SECONDS:
                need = RESERVE_MIB + min(estimate_bytes(e) for e in pending) / MIB
                if defer is not None:
                    for event in pending:
                        defer(
                            event, RESERVE_MIB + estimate_bytes(event) / MIB, free / MIB
                        )
                    pending.clear()
                    continue
                raise RuntimeError(
                    f"Memory budget requires at least {need:.0f} MiB available; "
                    f"currently {free / MIB:.0f} MiB. Resume this output when memory is available."
                )
            time.sleep(1)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return {"peak_active": peak, "minimum_available_mib": smallest_available / MIB}
