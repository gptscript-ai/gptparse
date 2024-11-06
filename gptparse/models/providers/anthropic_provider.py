from typing import List, Any
from anthropic import Anthropic
import base64
from ..types import Message, LLMResponse, TokenUsage, MessageContent
from ..base import LLMProvider, retry_with_backoff


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(
        self,
        model: str,
        temperature: float = None,
        max_tokens: int = None,
        system: str = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.client = Anthropic()
        self.system = system

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        converted_messages = []

        for message in messages:
            if isinstance(message["content"], str):
                converted_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )
            else:
                content_parts = []
                for part in message["content"]:
                    if part["type"] == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image":
                        image_source = part["source"]
                        if image_source["type"] == "base64":
                            content_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image_source["media_type"],
                                        "data": image_source["data"],
                                    },
                                }
                            )
                        elif image_source["type"] == "url":
                            # Convert URL to base64 for Anthropic
                            import requests

                            response = requests.get(image_source["data"])
                            image_data = base64.b64encode(response.content).decode()
                            content_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_data,
                                    },
                                }
                            )
                converted_messages.append(
                    {"role": message["role"], "content": content_parts}
                )

        return converted_messages

    @retry_with_backoff()  # Uses default from provider config
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._convert_messages(messages),
            **kwargs,
        }

        if self.system:
            request_params["system"] = self.system

        response = self.client.messages.create(**request_params)

        return LLMResponse(
            content=response.content[0].text if response.content else "",
            raw_response=response,
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                prediction_tokens=0,
            ),
        )
