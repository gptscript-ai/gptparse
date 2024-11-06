from typing import List, Union, Any
import google.generativeai as genai
from PIL import Image
from ..types import Message, LLMResponse, TokenUsage
from ..base import LLMProvider, retry_with_backoff


class GoogleProvider(LLMProvider):
    provider_name = "google"

    def __init__(
        self, model: str, temperature: float = None, max_tokens: int = None, **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

    def _convert_messages(
        self, messages: List[Message]
    ) -> Union[str, List[Union[str, Image.Image]]]:
        if len(messages) == 1 and isinstance(messages[0]["content"], str):
            return messages[0]["content"]

        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                converted.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        converted.append(item["text"])
                    elif item["type"] == "image":
                        image_source = item["source"]
                        if isinstance(image_source, Image.Image):
                            converted.append(image_source)
                        elif isinstance(image_source, dict):
                            if image_source["type"] == "url":
                                import requests
                                from io import BytesIO

                                response = requests.get(image_source["data"])
                                converted.append(Image.open(BytesIO(response.content)))

        return converted if len(converted) > 1 else converted[0]

    @retry_with_backoff()
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        prompt = self._convert_messages(messages)
        prompt_tokens = self.model.count_tokens(prompt)

        response = self.model.generate_content(prompt, **kwargs)

        return LLMResponse(
            content=response.text,
            raw_response=response,
            usage=TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count,
                completion_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count,
                prediction_tokens=0,
            ),
        )
