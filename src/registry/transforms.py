from typing import Callable, Dict
import numpy as np

_TRANSFORMS: Dict[str, Callable] = {}

def register_transform(name: str):
    def decorator(fn):
        key = name.upper()

        if key in _TRANSFORMS:
            raise ValueError(f"Transform '{name}' already registered")

        _TRANSFORMS[key] = fn
        return fn

    return decorator

def get_transform(name: str, n: int, **kwargs):
    if name.upper().startswith("LEARNED"):
        base = "LEARNED"

        if ":" in name:
            _, spec = name.split(":", 1)

            if spec.startswith("path="):
                kwargs["path"] = spec.split("=", 1)[1]

        key = base
    else:
        key = name.upper()

    if key not in _TRANSFORMS:
        raise ValueError(f"Unknown transform: {name}")

    fn = _TRANSFORMS[key]

    return fn(n, **kwargs)

def list_transforms():
    return list(_TRANSFORMS.keys())