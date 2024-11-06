import pytest
from gptparse.models.providers.openai_provider import OpenAIProvider
from gptparse.models.providers.anthropic_provider import AnthropicProvider
from gptparse.models.types import LLMResponse
from gptparse.models.providers.google_provider import GoogleProvider


@pytest.fixture
def provider():
    return OpenAIProvider(model="gpt-4o", temperature=0.01, max_tokens=100)


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider(
        model="claude-3-sonnet-20240229", temperature=0.01, max_tokens=100
    )


@pytest.fixture
def google_provider():
    return GoogleProvider(model="gemini-1.5-flash", temperature=0.01, max_tokens=100)


@pytest.mark.asyncio
async def test_openai_basic_completion(provider):
    messages = [{"role": "user", "content": "Hello"}]
    response = await provider.complete(messages)

    print("\n=== Basic Completion Test ===")
    print(f"Response content: {response.content}")
    print(f"Usage stats: {response.usage}")

    assert isinstance(response, LLMResponse)
    assert response.content and isinstance(response.content, str)
    assert isinstance(response.usage, dict)
    assert all(
        key in response.usage
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    )


@pytest.mark.asyncio
async def test_openai_with_image_input(provider):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "data": "https://developer-blogs.nvidia.com/wp-content/uploads/2020/07/OpenAI-GPT-3-featured-image.png",
                    },
                },
            ],
        }
    ]

    response = await provider.complete(messages)

    print("\n=== Image Input Test ===")
    print(f"Response content: {response.content}")
    print(f"Usage stats: {response.usage}")

    assert isinstance(response, LLMResponse)
    assert response.content and isinstance(response.content, str)


@pytest.mark.asyncio
async def test_anthropic_basic_completion(anthropic_provider):
    messages = [{"role": "user", "content": "Hello"}]
    response = await anthropic_provider.complete(messages)

    print("\n=== Anthropic Basic Completion Test ===")
    print(f"Response content: {response.content}")
    print(f"Usage stats: {response.usage}")

    assert isinstance(response, LLMResponse)
    assert response.content and isinstance(response.content, str)
    assert isinstance(response.usage, dict)
    assert all(
        key in response.usage
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    )


@pytest.mark.asyncio
async def test_anthropic_with_image_input(anthropic_provider):
    import base64
    import requests

    # Download and encode image
    image_url = "https://developer-blogs.nvidia.com/wp-content/uploads/2020/07/OpenAI-GPT-3-featured-image.png"
    image_data = requests.get(image_url).content
    base64_image = base64.b64encode(image_data).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                },
            ],
        }
    ]

    response = await anthropic_provider.complete(messages)

    print("\n=== Anthropic Image Input Test ===")
    print(f"Response content: {response.content}")
    print(f"Usage stats: {response.usage}")

    assert isinstance(response, LLMResponse)
    assert response.content and isinstance(response.content, str)


@pytest.mark.asyncio
async def test_google_basic_completion(google_provider):
    messages = [{"role": "user", "content": "Hello"}]
    response = await google_provider.complete(messages)

    print("\n=== Google Basic Completion Test ===")
    print(f"Response content: {response.content}")
    print(f"Usage stats: {response.usage}")

    assert isinstance(response, LLMResponse)
    assert response.content and isinstance(response.content, str)
    assert isinstance(response.usage, dict)
    assert all(
        key in response.usage
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    )
    # Add specific token count assertions
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0
    assert response.usage["total_tokens"] == (
        response.usage["prompt_tokens"] + response.usage["completion_tokens"]
    )


@pytest.mark.asyncio
async def test_google_with_image_input(google_provider):
    import requests
    import PIL.Image
    from io import BytesIO

    # Download and process image
    image_url = "https://developer-blogs.nvidia.com/wp-content/uploads/2020/07/OpenAI-GPT-3-featured-image.png"
    image_data = requests.get(image_url).content

    # Convert to PIL Image for Google's API
    pil_image = PIL.Image.open(BytesIO(image_data))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "source": pil_image},
            ],
        }
    ]

    response = await google_provider.complete(messages)

    print("\n=== Google Image Input Test ===")
    print(f"Response content: {response.content}")
    print(f"Usage stats: {response.usage}")

    assert isinstance(response, LLMResponse)
    assert response.content and isinstance(response.content, str)
    assert isinstance(response.usage, dict)
    assert all(
        key in response.usage
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    )
