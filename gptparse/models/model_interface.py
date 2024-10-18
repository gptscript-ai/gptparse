from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any

PROVIDER_MODELS = {
    "openai": {"default": "gpt-4o", "options": ["gpt-4o", "gpt-4o-mini"]},
    "anthropic": {
        "default": "claude-3-5-sonnet-20240620",
        "options": [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
    "google": {
        "default": "gemini-1.5-pro-002",
        "options": [
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b",
        ],
    },
}


def get_model(provider: str, model: str = None, **kwargs: Dict[str, Any]):
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unsupported provider: {provider}")

    if model is None:
        model = PROVIDER_MODELS[provider]["default"]
    elif model not in PROVIDER_MODELS[provider]["options"]:
        raise ValueError(f"Unsupported model for {provider}: {model}")

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            **kwargs,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            **kwargs,
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(model=model, temperature=0, **kwargs)


def list_available_models(provider: str = None):
    if provider:
        if provider not in PROVIDER_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        return PROVIDER_MODELS[provider]["options"]
    else:
        return {p: models["options"] for p, models in PROVIDER_MODELS.items()}
