import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "osmesa")

from mujoco_module.meshes import load_pieces, parse_mesh
from mujoco_module.sim import SimManager, WorldConfig

BLOCK = {
    "name": "pick_block",
    "type": "cube",
    "size": 0.0313,
    "position": [0.3, 0.0, 0.01565],
    "mass": 0.03,
    "static_friction": 2.0,
    "dynamic_friction": 2.0,
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sim(clock: FakeClock) -> SimManager:
    """A SimManager driven inline on the test thread: run() executes
    immediately and nothing steps unless the test calls advance() or tick()."""
    manager = SimManager(clock=clock)
    manager.bind_current_thread()
    manager.configure_world(WorldConfig.from_attrs({"props": [BLOCK]}))
    yield manager
    manager._close_renderers()


@pytest.fixture
def running_sim(monkeypatch: pytest.MonkeyPatch) -> SimManager:
    """A SimManager whose main_loop runs on a background thread, as in
    production, installed as the instance component models get."""
    manager = SimManager()
    thread = threading.Thread(target=manager.main_loop, daemon=True)
    thread.start()
    monkeypatch.setattr(SimManager, "_instance", manager)
    yield manager
    manager.request_stop()
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def u_pieces(tmp_path_factory):
    """The bundled bin decomposed once for the session (CoACD takes seconds)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VIAM_MODULE_DATA", str(tmp_path_factory.mktemp("cache")))
        return load_pieces("bin", parse_mesh("bin", "meshes/bin.stl"))
