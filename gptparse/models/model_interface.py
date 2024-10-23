import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any

PROVIDER_MODELS = {
    "openai": {
        "default": "gpt-4o",
        "options": ["gpt-4o", "gpt-4o-mini"],
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "default": "claude-3-5-sonnet-latest",
        "options": [
            "claude-3-5-sonnet-latest",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "env_var": "ANTHROPIC_API_KEY",
    },
    "google": {
        "default": "gemini-1.5-pro-002",
        "options": [
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b",
        ],
        "env_var": "GOOGLE_API_KEY",
    },
}


def check_api_key(provider: str):
    env_var = PROVIDER_MODELS[provider]["env_var"]
    if not os.getenv(env_var):
        raise ValueError(
            f"The {env_var} environment variable is not set. "
            f"Please set it to use the {provider} provider."
        )


def get_model(provider: str, model: str = None, **kwargs: Dict[str, Any]):
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unsupported provider: {provider}")

    check_api_key(provider)

    if model is None:
        model = PROVIDER_MODELS[provider]["default"]
    elif model not in PROVIDER_MODELS[provider]["options"]:
        raise ValueError(f"Unsupported model for {provider}: {model}")

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0.01,
            max_tokens=4096,
            timeout=None,
            max_retries=2,
            **kwargs,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=0.01,
            max_tokens=4096,
            timeout=None,
            max_retries=2,
            **kwargs,
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=model, temperature=0.01, max_tokens=4096, max_retries=2, **kwargs
        )


def list_available_models(provider: str = None):
    if provider:
        if provider not in PROVIDER_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        return PROVIDER_MODELS[provider]["options"]
    else:
        return {p: models["options"] for p, models in PROVIDER_MODELS.items()}
