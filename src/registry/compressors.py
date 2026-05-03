from typing import Type, Dict

_COMPRESSORS: Dict[str, Type] = {}

def register_compressor(name: str):
    def decorator(cls):
        key = name.upper()

        if key in _COMPRESSORS:
            raise ValueError(f"Compressor '{name}' already registered")

        _COMPRESSORS[key] = cls
        return cls

    return decorator

def get_compressor(name: str):
    key = name.upper()

    if key not in _COMPRESSORS:
        raise ValueError(f"Unknown compressor: {name}")

    return _COMPRESSORS[key]

def list_compressors():
    return list(_COMPRESSORS.keys())