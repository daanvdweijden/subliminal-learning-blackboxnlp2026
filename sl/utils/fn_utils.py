from typing import TypeVar
from functools import wraps
import time
import random
import asyncio

from loguru import logger

S = TypeVar("S")
T = TypeVar("T")


def max_concurrency_async(max_size: int):
    """
    Decorator that limits the number of concurrent executions of an async function using a semaphore.

    Args:
        max_size: Maximum number of concurrent executions allowed

    Returns:
        Decorated async function with concurrency limiting
    """
    import asyncio

    def decorator(func):
        semaphore = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def auto_retry(exceptions: list[type[Exception]], max_retry_attempts: int = 3):
    """
    Decorator that retries function calls with exponential backoff on specified exceptions.

    Args:
        exceptions: List of exception types to retry on
        max_retry_attempts: Maximum number of retry attempts (default: 3)

    Returns:
        Decorated function that automatically retries on specified exceptions
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if attempt == max_retry_attempts:
                        raise e

                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def auto_retry_async(
    exceptions: list[type[Exception]],
    max_retry_attempts: int = 3,
    log_exceptions: bool = False,
):
    """
    Decorator that retries async function calls with exponential backoff on specified exceptions.

    Args:
        exceptions: List of exception types to retry on
        max_retry_attempts: Maximum number of retry attempts (default: 3)

    Returns:
        Decorated async function that automatically retries on specified exceptions
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if log_exceptions:
                        logger.exception(e)
                    if attempt == max_retry_attempts:
                        raise e
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)

            logger.warning(f"last attempt of {func.__name__}")
            return await func(*args, **kwargs)

        return wrapper

    return decorator
