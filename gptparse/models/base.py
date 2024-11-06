from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypeVar
import asyncio
import random
import functools
from dataclasses import dataclass
from .types import LLMResponse

T = TypeVar("T")


@dataclass
class SessionConfig:
    """Configuration for API session management"""

    timeout: int = 60
    max_retries: int = 3
    backoff_factor: float = 1.0


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_retries: int = 3
    min_seconds: float = 1.0
    max_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded"""

    pass


class InvalidRequestError(Exception):
    """Raised when the API request is invalid"""

    pass


def retry_with_backoff(retry_config: Optional[RetryConfig] = None):
    """Decorator for retrying API calls with exponential backoff"""
    if retry_config is None:
        retry_config = RetryConfig()

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(retry_config.max_retries):
                try:
                    return await func(self, *args, **kwargs)
                except (RateLimitError, InvalidRequestError) as e:
                    last_exception = e
                    if attempt == retry_config.max_retries - 1:
                        raise

                    backoff = min(
                        retry_config.max_seconds,
                        retry_config.min_seconds
                        * (retry_config.exponential_base**attempt),
                    )

                    if retry_config.jitter:
                        backoff = backoff * (0.5 + random.random())

                    await asyncio.sleep(backoff)

            raise last_exception

        return wrapper

    return decorator


class LLMProvider(ABC):
    """Base class for language model providers"""

    provider_name: str = None

    def __init__(
        self,
        model: str,
        temperature: float = 0.01,
        max_tokens: int = 4096,
        session_config: Optional[SessionConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_config = session_config or SessionConfig()
        self.retry_config = retry_config or RetryConfig()
        self.extra_kwargs = kwargs

    @abstractmethod
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> Any:
        """Convert messages to provider-specific format"""
        pass

    @abstractmethod
    @retry_with_backoff()
    async def complete(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Send a completion request to the model"""
        pass

    async def batch_complete(
        self,
        messages_list: List[List[Dict[str, Any]]],
        max_concurrency: int = 5,
        callback=None,
    ) -> List[Any]:
        """Process a batch of completion requests with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_with_semaphore(messages):
            async with semaphore:
                try:
                    response = await self.complete(messages)
                    if callback:
                        callback.on_completion()
                    return response
                except Exception as e:
                    if callback:
                        callback.on_error(e)
                    raise

        tasks = [process_with_semaphore(messages) for messages in messages_list]
        return await asyncio.gather(*tasks)
