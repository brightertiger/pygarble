"""Shared configuration and batch contracts."""

import math
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from typing import Any, Callable, List, Optional, TypeVar, Union

T = TypeVar("T")


def finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be non-negative (an integer)")
    return value


def parameter_value(name: str, value: object, default: object) -> Any:
    (
        "Validate legacy numeric kwargs before strategies do the"
        "ir range checks."
    )
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be at least 1 (a positive integer)")
        return value
    return finite_number(name, value)


def unit_interval(name: str, value: object) -> float:
    number = finite_number(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return number


def validate_batch(texts: List[str]) -> None:
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"all elements must be strings; element {i} is "
                f"{type(text).__name__}"
            )


def process_input(
    texts: Union[str, List[str]],
    function: Callable[[str], T],
    threads: Optional[int],
    timeout: Optional[float] = None,
    max_input_length: Optional[int] = None,
) -> Union[T, List[T]]:
    if isinstance(texts, str):
        batch = [texts]
    elif isinstance(texts, list):
        validate_batch(texts)
        batch = texts
    else:
        raise TypeError("Input must be a string or list of strings")
    if max_input_length is not None and any(
        len(text) > max_input_length for text in batch
    ):
        raise ValueError("text exceeds max_input_length")
    if isinstance(texts, str):
        return function(texts)
    if not threads or threads == 1 or len(batch) < 10:
        return [function(text) for text in batch]
    # Bounded submission keeps worker memory independent of batch size.
    # Timeouts propagate; Python threads cannot enforce a hard deadline.
    results: List[T] = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for start in range(0, len(batch), threads * 2):
            futures = [
                executor.submit(function, text)
                for text in batch[start : start + threads * 2]
            ]
            results.extend(
                future.result(timeout=timeout) for future in futures
            )
    return results
