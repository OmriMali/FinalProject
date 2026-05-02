from src.core.registry import make_registry

COMPRESSORS, register_compressor, get_compressor, list_compressors = make_registry()
