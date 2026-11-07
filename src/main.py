"""Module entrypoint.

The sim thread owns MuJoCo (see sim.py), so the layout is:
  - main thread: SimManager.main_loop() - steps the simulator
  - side thread: the Viam module gRPC server

When viam-server shuts the module down, the module thread exits and asks the
sim loop to stop.
"""

import asyncio
import signal
import sys
import threading

from viam.logging import getLogger
from viam.module.module import Module

import mujoco_module.models  # noqa: F401  (registers the models)
from mujoco_module.sim import SimManager

LOGGER = getLogger("viam-mujoco-sim.main")


def _run_module(sim: SimManager) -> None:
    loop = asyncio.new_event_loop()
    # grpclib wants to install signal handlers, which only works on the main
    # thread - and the main thread belongs to the sim. main() handles signals.
    loop.add_signal_handler = lambda *args, **kwargs: None  # type: ignore[method-assign]
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(Module.run_from_registry())
    except Exception:
        LOGGER.exception("module server exited with error")
    finally:
        sim.request_stop()


def main() -> None:
    sim = SimManager.get()

    def _shutdown(signum, frame):
        sim.request_stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    thread = threading.Thread(target=_run_module, args=(sim,), name="viam-module", daemon=True)
    thread.start()
    sim.main_loop()
    sys.exit(0)


if __name__ == "__main__":
    main()
