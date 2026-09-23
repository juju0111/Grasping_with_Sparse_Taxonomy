"""RL training infra: registry + base classes + concrete tasks.

Importing this package auto-imports every concrete task module under
``rl_envs/``, populating the env registry so ``make_rl_env('<name>', ...)``
works without the caller having to import the task module by hand.

Order matters: ``rl_env_base`` must load **first** (it defines the
registry, the base classes, and ``register_rl_env``); each task module
then attaches itself to that registry via the decorator.
"""
# Load the framework primitives first — tasks inherit from / reference these.
from . import rl_env_base    # noqa: F401

# Then eager-import every registered task. Side effect: each task module's
# ``@register_rl_env(...)`` decorator runs and adds an entry to
# ``rl_env_base._RL_ENV_REGISTRY``.
from . import rl_envs         # noqa: F401
