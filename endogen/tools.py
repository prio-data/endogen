from typing import List, Iterable, Any, Sequence
from functools import wraps
from time import time


def measure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f"Elapsed time: {end-start}")
        return result

    return wrapper


def flatten_recursive_generator(lst: List[Any]) -> Iterable[Any]:
    """Flatten a list using recursion."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten_recursive(item)
        else:
            yield item


def flatten(list_of_lists: Sequence) -> Sequence:
    return [item for sublist in list_of_lists for item in sublist]


def flatten_recursive(list_of_lists: Sequence) -> Sequence:
    return list(flatten_recursive_generator(list_of_lists))
