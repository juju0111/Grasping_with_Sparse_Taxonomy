"""Policy network module — eager-import every policy file so their
``@register_policy(...)`` decorators run at package import time.

After ``import grit.model`` (or any submodule import that triggers this
``__init__``), ``make_policy('<name>', ...)`` can find every registered
policy without the caller having to import each policy file by hand.
"""
# Load the registry first.
from . import base_networks                 # noqa: F401

# Each task-specific policy module registers itself via @register_policy.
from . import shared_cross_embodiment       # noqa: F401
from . import shared_single                  # noqa: F401
from . import bps_networks                   # noqa: F401
