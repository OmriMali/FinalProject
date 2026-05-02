def make_registry():
    registry = {}

    def register(name):
        key = name.upper()

        def decorator(obj):
            if key in registry:
                raise ValueError(f"{name} already registered")
            registry[key] = obj
            return obj

        return decorator

    def get(name):
        key = name.upper()
        if key not in registry:
            raise ValueError(f"Unknown key: {name}")
        return registry[key]

    def list_keys():
        return list(registry.keys())

    return registry, register, get, list_keys