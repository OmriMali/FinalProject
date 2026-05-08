from src.compressors.base import Compressor



_COMPRESSORS: dict[str, type[Compressor]] = {}

def register_compressor(cls: type[Compressor]) -> type[Compressor]:
    """
    Register a compressor class by its class-level name.

    Parameters
    ----------
    cls : type[Compressor]
        Compressor class to register.

    Returns
    -------
    type[Compressor]
        Registered compressor class.

    Raises
    ------
    ValueError
        If the compressor has no name or if the name is already registered.
    """

    name = getattr(cls, "name", None)

    if not name:
        raise ValueError("Compressor class must define a non-empty name")

    if name in _COMPRESSORS:
        raise ValueError(f"Compressor already registered: {name}")

    _COMPRESSORS[name] = cls

    return cls

def get_compressor(name: str) -> type[Compressor]:
    """
    Get a registered compressor class by name.

    Parameters
    ----------
    name : str
        Registered compressor name.

    Returns
    -------
    type[Compressor]
        Compressor class.

    Raises
    ------
    KeyError
        If no compressor is registered with the given name.
    """

    if name not in _COMPRESSORS:
        raise KeyError(f"Unknown compressor: {name}")

    return _COMPRESSORS[name]

def list_compressors():
    """
    Return the names of all registered compressors.

    Returns
    -------
    list[str]
        Registered compressor names.
    """
    return sorted(_COMPRESSORS.keys())