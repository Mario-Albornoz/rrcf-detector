def hash_based_partitioner(
    exchange: str, instrument_class: str, num_workers: int
) -> int:
    """Deterministic key-to-worker assignment using hash modulo."""
    key = f"{exchange}:{instrument_class}"
    return hash(key) % num_workers
