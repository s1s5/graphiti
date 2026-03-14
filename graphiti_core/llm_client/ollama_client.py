"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI
else:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            'openai is required for OllamaClient. Install it with: pip install graphiti-core[openai]'
        ) from None

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'llama3'
DEFAULT_MAX_TOKENS = 2048
DEFAULT_BASE_URL = 'http://localhost:11434/v1'


class OllamaConfig(LLMConfig):
    """Configuration for Ollama client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
    ):
        # Ollama uses "ollama" as default api key
        super().__init__(
            api_key=api_key or 'ollama',
            model=model or DEFAULT_MODEL,
            base_url=base_url or DEFAULT_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            small_model=small_model,
        )


class OllamaClient(LLMClient):
    """Ollama LLM Client

    This client connects to a local Ollama instance using OpenAI-compatible API.
    """

    def __init__(self, config: OllamaConfig | None = None, cache: bool = False):
        if config is None:
            config = OllamaConfig()
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
        super().__init__(config, cache)

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        # Convert messages to OpenAI format
        msgs: list[dict] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})

        # Determine which model to use based on model_size
        model = self.small_model if model_size == ModelSize.small else self.model
        model = model or DEFAULT_MODEL

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                # Note: response_format not supported by Ollama in all versions
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
