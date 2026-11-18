"""Send Python and MuJoCo warnings to the module's log at WARN.

viam-server logs every line a module writes to stderr at ERROR (and a
MuJoCo warning's trailing blank line as an empty ERROR), which reads as a
failure to anyone grepping the log."""

import logging
import warnings

import mujoco


def route_warnings(logger: logging.Logger) -> None:
    def show(message, category, filename, lineno, file=None, line=None):
        logger.warning(warnings.formatwarning(message, category, filename, lineno, line).strip())

    warnings.showwarning = show
    mujoco.set_mju_user_warning(lambda text: logger.warning(f"MuJoCo: {text.strip()}"))
