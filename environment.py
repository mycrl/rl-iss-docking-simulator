"""
Compatibility shim — the environment has moved to the ``docking`` package.

Use ``from docking import IssDockingEnv`` in new code.
"""

from docking.environment import IssDockingEnv  # noqa: F401

__all__ = ["IssDockingEnv"]
