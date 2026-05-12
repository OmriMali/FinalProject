from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path


def format_shape(shape) -> str:
    """
    Format a shape tuple as plain Python integers.
    """

    return str(tuple(int(x) for x in shape))


def format_config_value(value):
    """
    Convert config values to readable CSV-friendly values.
    """

    if isinstance(value, Enum):
        return value.name

    if isinstance(value, str) and value.startswith("LEARNED:path="):
        path = value.split("LEARNED:path=", 1)[1]
        return Path(path).name

    if isinstance(value, tuple):
        return tuple(format_config_value(v) for v in value)

    if isinstance(value, list):
        return [format_config_value(v) for v in value]

    return value


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
        value = format_config_value(value)

        if key == "sr" and isinstance(value, tuple):
            row["sr_h"] = value[0]
            row["sr_w"] = value[1]
            row["sr_b"] = value[2]

        elif key == "Phis" and isinstance(value, tuple):
            row["Phi_h"] = value[0]
            row["Phi_w"] = value[1]
            row["Phi_b"] = value[2]

        elif key == "Psis" and isinstance(value, tuple):
            row["Psi_h"] = value[0]
            row["Psi_w"] = value[1]
            row["Psi_b"] = value[2]

        else:
            row[key] = value

    return row