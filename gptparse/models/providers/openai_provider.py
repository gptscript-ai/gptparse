from typing import List, Optional
from openai import OpenAI
from ..types import Message, LLMResponse, TokenUsage
from ..base import LLMProvider, retry_with_backoff


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(
        self, model: str, temperature: float = None, max_tokens: int = None, **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.client = OpenAI()

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
                        content_parts.append(
                            {"type": "text", "text": part["text"]}
                        )
                    elif part["type"] == "image":
                        image_source = part["source"]
                        if image_source["type"] in ["url", "base64"]:
                            content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_source["data"]},
                                }
                            )
                converted_messages.append(
                    {"role": message["role"], "content": content_parts}
                )

        return converted_messages

    @retry_with_backoff()
    async def complete(
        self, messages: List[Message], prediction: Optional[dict] = None, **kwargs
    ) -> LLMResponse:
        completion_kwargs = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        if prediction:
            completion_kwargs["prediction"] = prediction

        completion = self.client.chat.completions.create(**completion_kwargs)

        prediction_tokens = (
            completion.usage.completion_tokens_details.accepted_prediction_tokens
            if hasattr(completion.usage, "completion_tokens_details")
            else 0
        )

        return LLMResponse(
            content=completion.choices[0].message.content,
            raw_response=completion,
            usage=TokenUsage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
                prediction_tokens=prediction_tokens,
            ),
        )
