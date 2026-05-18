def normalize_sequence_index(index: int, size: int, *, collection_name: str) -> int:
    normalized_index = index + size if index < 0 else index
    if normalized_index < 0 or normalized_index >= size:
        raise IndexError(
            f"{collection_name} index {index} out of range for {size} items"
        )
    return normalized_index
