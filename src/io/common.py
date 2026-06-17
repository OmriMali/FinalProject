from pathlib import Path

def ensure_npz_name(name: str) -> str:
    """
    Return filename with an '.npz' suffix.
    """
    if name.endswith(".npz"):
        return name
    return f"{name}.npz"

def make_npz_path(directory: str | Path, name: str) -> Path:
    """
    Build an '.npz' file path and create the directory if needed.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    return directory / ensure_npz_name(name)

def resolve_npz_path(directory: str | Path, name: str) -> Path:
    """
    Build an existing '.npz' file path for loading.
    """
    return Path(directory) / ensure_npz_name(name)


def list_npz_files(directory: str | Path) -> list[Path]:
    """
    List all ``.npz`` files in a directory.
    """
    return sorted(Path(directory).glob("*.npz"))