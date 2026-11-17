"""Reuse a frozen, void host launcher; only its runtime arguments may change."""

from functools import wraps
from inspect import signature
from threading import Lock

import flydsl.compiler as flyc


def compiled_launch(jit_launch):
    parameters = signature(jit_launch.func)
    lock = Lock()
    compiled = None

    @wraps(jit_launch.func)
    def launch(*args, **kwargs):
        nonlocal compiled
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        values = tuple(bound.arguments.values())
        if compiled is not None:
            return compiled(*values)
        with lock:
            if compiled is None:
                compiled = flyc.compile(jit_launch, *bound.args, **bound.kwargs)
                return None
        return compiled(*values)

    return launch
