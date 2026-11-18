"""Warnings reach the module's log, not stderr (viam-server logs every stderr line as ERROR)."""

import logging
import warnings

import mujoco

from mujoco_module.logs import route_warnings


def test_python_and_mujoco_warnings_go_to_the_logger(caplog, capfd):
    logger = logging.getLogger("test-route-warnings")
    before = mujoco.get_mju_user_warning(), warnings.showwarning
    try:
        route_warnings(logger)
        with caplog.at_level(logging.WARNING, logger=logger.name), warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("invalid value encountered in cast", RuntimeWarning)
            mujoco.get_mju_user_warning()("ARB_clip_control unavailable")
    finally:
        mujoco.set_mju_user_warning(before[0])
        warnings.showwarning = before[1]
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("RuntimeWarning: invalid value encountered in cast" in m for m in messages)
    assert "MuJoCo: ARB_clip_control unavailable" in messages
    assert capfd.readouterr().err == ""
