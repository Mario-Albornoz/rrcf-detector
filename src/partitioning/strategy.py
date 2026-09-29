def hash_based_partitioner(
    exchange: str, instrument_class: str, num_workers: int
) -> int:
    key = f"{exchange}:{instrument_class}"
    return hash(key) % num_workers
