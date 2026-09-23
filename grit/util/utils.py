from termcolor import colored
import numpy as np

def print_red(str):
    """
    Print a string in red color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'red'))
    
def print_yellow(str):
    """
    Print a string in yellow color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'yellow'))    

def print_green(str):
    """
    Print a string in green color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'green'))   
    
def print_blue(str):
    """
    Print a string in blue color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'blue'))    

def print_light_green(str):
    """
    Print a string in light green color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'light_green'))    

def print_light_blue(str):
    """
    Print a string in light blue color.
    
    Parameters:
        str (str): String to print.
    
    Returns:
        None
    """
    print (colored(str,'light_blue'))    


def quat2r(q, order='wxyz'):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Parameters:
        q (np.array): Quaternion in the form [w, x, y, z].
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """    
    if order == 'wxyz':
        w, x, y, z = q
    elif order == 'xyzw':
        x, y, z, w = q
    else:
        raise ValueError(f"Invalid order: {order}")
        
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def rpy2r(rpy_rad):
    """
    Convert roll, pitch, and yaw angles (in radians) to a 3x3 rotation matrix.
    
    Parameters:
        rpy_rad (np.array): Array of [roll, pitch, yaw] in radians.
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.cos(roll)
    Sphi  = np.sin(roll)
    Cthe  = np.cos(pitch)
    Sthe  = np.sin(pitch)
    Cpsi  = np.cos(yaw)
    Spsi  = np.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R


# ──────────────────────────────────────────────────────────────────────────
# Generic dict / string helpers
# ──────────────────────────────────────────────────────────────────────────

def fmt_template(s, **vars) -> str:
    """Apply :py:meth:`str.format` with ``**vars`` and fall back to the
    original string on any missing key. Non-strings are returned unchanged
    so callers can pass yaml values blindly.

    Used by ``scripts/train.py`` for ``wandb.name_template`` /
    ``wandb.group_template`` substitution (``{hand_name}`` / ``{task_name}`` /
    ``{save_dir_name}`` / …).
    """
    if not isinstance(s, str):
        return s
    try:
        return s.format(**vars)
    except (KeyError, IndexError):
        return s


def flatten_dict_with_prefix(prefix: str, d: dict, out: dict) -> None:
    """Flatten a nested dict so every leaf becomes a ``prefix + dotted/slashed
    path`` key in ``out`` (in-place). Useful for sending nested cfg blocks
    to flat metric sinks (W&B ``config=...``, CSV columns, ...).

    Example::

        out = {}
        flatten_dict_with_prefix("handler/", {"W_POS": 5.0, "gates": {"TOUCH_EPS": 1e-4}}, out)
        # out == {"handler/W_POS": 5.0, "handler/gates/TOUCH_EPS": 1e-4}
    """
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict_with_prefix(f"{prefix}{k}/", v, out)
        else:
            out[f"{prefix}{k}"] = v
