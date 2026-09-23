"""Task modules — importing a module runs its ``@register_rl_env`` registration.

* :mod:`grasping_core`    — the shared task (no env registered)
* :mod:`grasping_teacher` — ``grasping_teacher``: privileged teacher
* :mod:`grasping_student` — ``grasping_student``: deployable student
"""
from . import grasping_teacher   # noqa: F401
from . import grasping_student   # noqa: F401
