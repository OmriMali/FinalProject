from typing import Dict, Any
from src import util


def make_registry(configurable: bool = False):
    registry: Dict[str, Any] = {}

    # Register decorator
    def register(name: str):
        key = name.upper()

        def decorator(obj):
            if key in registry:
                raise ValueError(f"{name} already registered")
            registry[key] = obj
            return obj

        return decorator


    # Simple get
    def get(name: str, **kwargs):
        key = name.upper()
        if key not in registry:
            raise ValueError(f"Unknown key: {name}")

        obj = registry[key]

        return obj(**kwargs)

    # Config-aware get
    def get_configurable(name: str, *args, **kwargs):
        base_name, parsed = util.parse_config_string(name)
        parsed.update(kwargs)

        key = base_name.upper()

        if key not in registry:
            raise ValueError(f"Unknown key: {name}")

        fn = registry[key]
        return fn(*args, **parsed)

    # List
    def list_keys():
        return list(registry.keys())


    # Choose correct getter
    if configurable:
        return registry, register, get_configurable, list_keys
    else:
        return registry, register, get, list_keys