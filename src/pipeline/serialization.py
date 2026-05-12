from dataclasses import asdict, is_dataclass
from enum import Enum


def config_to_row(config) -> dict:
    """
    Convert a config object to flat CSV-friendly fields.
    """

    if is_dataclass(config):
        data = asdict(config)
    else:
        data = dict(config.__dict__)

    row = {}

    for key, value in data.items():
        if isinstance(value, Enum):
            row[key] = value.name
        elif isinstance(value, tuple):
            row[key] = str(value)
        elif isinstance(value, list):
            row[key] = str(value)
        else:
            row[key] = value

    return row