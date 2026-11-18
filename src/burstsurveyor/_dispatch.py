"""Memory-aware scheduling. Only the coordinator commits completed worker output."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import psutil

from ._storage import read, write

MIB = 1024**2
RESERVE_MIB = 512
MAX_WORKERS = 3
MEMORY_WAIT_SECONDS = 30
EVENT_WATCHDOG_SECONDS = 600


def _write_worker_result(result_path, work, args):
    configure_worker()
    write(result_path, work(*args))


def run_isolated(work, args, *, scratch, timeout=EVENT_WATCHDOG_SECONDS):
    """Run one event in a killable process. Never return a partial result."""
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        result_path = Path(temporary) / "result.json"
        ctx = multiprocessing.get_context("spawn")
        process = ctx.Process(target=_write_worker_result, args=(result_path, work, args))
        process.start()
        try:
            process.join(timeout)
            if process.is_alive():
                raise TimeoutError(f"Emergency event watchdog exceeded {timeout:g} seconds")
            if process.exitcode != 0:
                raise RuntimeError(f"Event worker exited with code {process.exitcode}")
            return read(result_path)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(2)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()



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


def dispatch(events, workers, make_task, work, commit, *, defer=None, isolated=False):
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
                            # The bounded worker already supervises a child process per event.
                            pool = (
                                ThreadPoolExecutor(max_workers=workers)
                                if isolated else ProcessPoolExecutor(
                                    max_workers=workers,
                                    initializer=configure_worker,
                                    mp_context=multiprocessing.get_context("spawn"),
                                )
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
