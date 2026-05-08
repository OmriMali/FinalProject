from src.dictionary_trainers.base import DictionaryTrainer


_DICTIONARY_TRAINERS: dict[str, type[DictionaryTrainer]] = {}


def register_dictionary_trainer(cls: type[DictionaryTrainer],) -> type[DictionaryTrainer]:
    """
    Register a dictionary trainer class by its class-level name.

    Parameters
    ----------
    cls : type[DictionaryTrainer]
        Dictionary trainer class to register.

    Returns
    -------
    type[DictionaryTrainer]
        Registered dictionary trainer class.

    Raises
    ------
    ValueError
        If the trainer has no name or if the name is already registered.
    """
    name = getattr(cls, "name", None)

    if not name:
        raise ValueError(
            "Dictionary trainer class must define a non-empty name"
        )

    if name in _DICTIONARY_TRAINERS:
        raise ValueError(
            f"Dictionary trainer already registered: {name}"
        )

    _DICTIONARY_TRAINERS[name] = cls

    return cls


def get_dictionary_trainer(name: str) -> type[DictionaryTrainer]:
    """
    Get a registered dictionary trainer class by name.
    """
    if name not in _DICTIONARY_TRAINERS:
        raise KeyError(f"Unknown dictionary trainer: {name}")

    return _DICTIONARY_TRAINERS[name]


def available_dictionary_trainers() -> list[str]:
    """
    Return the names of all registered dictionary trainers.
    """
    return sorted(_DICTIONARY_TRAINERS.keys())