"""Simple retry utility with exponential backoff."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(
    operation: Callable[[], T],
    retries: int = 3,
    base_delay_seconds: float = 0.1,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> T:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return operation()
        except retry_on as error:
            last_error = error
            if attempt == retries - 1:
                break
            time.sleep(base_delay_seconds * (2**attempt))
    if last_error is None:
        raise RuntimeError("Retry failed without captured exception.")
    raise last_error

