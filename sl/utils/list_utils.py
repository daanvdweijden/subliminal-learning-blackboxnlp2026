from typing import TypeVar

T = TypeVar("T")


def flatten(lst: list[list[T]]) -> list[T]:
    """Flattens a list of lists into a single list."""
    return [item for sublist in lst for item in sublist]


def batch(lst: list[T], size: int) -> list[list[T]]:
    """Groups list into batches of specified size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]
