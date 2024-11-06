from typing import TypedDict, Literal, Union, List, Dict, Any
from dataclasses import dataclass, asdict
from PIL.Image import Image

class ImageSource(TypedDict):
    type: Literal["base64", "url", "pil"]
    media_type: str
    data: Union[str, bytes, Image]

class MessageContent(TypedDict):
    type: Literal["text", "image"]
    content: Union[str, ImageSource]

class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[MessageContent]]

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prediction_tokens: int = 0

    def to_dict(self):
        return asdict(self)

@dataclass
class LLMResponse:
    content: str
    raw_response: Any
    usage: TokenUsage

    def __post_init__(self):
        # Convert TokenUsage to dict for backward compatibility
        if isinstance(self.usage, TokenUsage):
            self.usage = self.usage.to_dict()