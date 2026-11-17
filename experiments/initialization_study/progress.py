"""Immediate stage diagnostics with elapsed time and periodic waiting messages."""

import traceback
from contextlib import contextmanager
from datetime import datetime
from threading import Event, Thread
from time import perf_counter
from collections.abc import Iterator


"""
    Prints timestamped progress immediately, including when stdout is redirected.
"""
def progress(message: str) -> None:
    print("[{}] {}".format(datetime.now().astimezone().isoformat(timespec="seconds"), message), flush=True)


"""
    Reports stage start, completion, failures, and a heartbeat every 30 seconds.
    A heartbeat indicates the stage has not returned; it does not guarantee GPU progress.
"""
@contextmanager
def stage(name: str, heartbeat_seconds: float = 30.0) -> Iterator[None]:
    started = perf_counter()
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(heartbeat_seconds):
            progress("WAIT {} | {:.1f}s elapsed; stage has not returned".format(name, perf_counter() - started))

    progress("START " + name)
    worker = Thread(target=heartbeat, daemon=True)
    worker.start()

    try:
        yield
    except BaseException:
        progress("FAILED {} | {:.1f}s".format(name, perf_counter() - started))
        traceback.print_exc()
        raise
    else:
        progress("DONE {} | {:.1f}s".format(name, perf_counter() - started))
    finally:
        stopped.set()
        worker.join()
