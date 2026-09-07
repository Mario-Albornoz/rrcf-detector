from src.kafka.consumer import NormalizedVectorDto


def get_instrument_key(data: NormalizedVectorDto) -> str:
    return f"{data.exchange}:{data.instrument_class}"
