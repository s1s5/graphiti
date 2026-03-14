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
import re
import typing
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI
else:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "openai is required for OllamaClient. Install it with: pip install graphiti-core[openai]"
        ) from None

from pydantic import BaseModel, ValidationError

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TEMPERATURE = 0.1  # Low temperature for structured output
DEFAULT_MAX_RETRIES = 10


class OllamaConfig(LLMConfig):
    """Configuration for Ollama client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
    ):
        # Ollama uses "ollama" as default api key
        super().__init__(
            api_key=api_key or "ollama",
            model=model or DEFAULT_MODEL,
            base_url=base_url or DEFAULT_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            small_model=small_model,
        )


class OllamaClient(LLMClient):
    """Ollama LLM Client

    This client connects to a local Ollama instance using OpenAI-compatible API.
    Supports structured output through JSON extraction and validation.
    """

    def __init__(
        self,
        config: OllamaConfig | None = None,
        cache: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if config is None:
            config = OllamaConfig()
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
        super().__init__(config, cache)

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.max_retries = max_retries

    def _extract_json(self, text: str) -> dict[str, typing.Any]:
        """Extract JSON from LLM response text.

        Handles various formats:
        - ```json ... ``` code blocks
        - ``` ... ``` code blocks
        - Raw JSON objects

        Args:
            text: Raw response text from LLM

        Returns:
            Parsed JSON as dictionary
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to extract from ```json ... ``` block
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to extract from any ``` ... ``` block
        code_match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: try parsing the whole text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        logger.warning(f"Failed to extract JSON from response: {text[:200]}...")
        return {}

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
            if m.role == "user":
                msgs.append({"role": "user", "content": m.content})
            elif m.role == "system":
                if response_model:
                    content = (
                        m.content
                        + f"\n\n以下のJSON形式で正確に回答してください:\n```json\n{response_model.model_json_schema()}\n```"
                    )
                msgs.append({"role": "system", "content": content})
        if response_model and msgs[-1]["role"] == "user":
            msgs[-1][
                "content"
            ] += f"\n\n以下のJSON形式で正確に回答してください:\n```json\n{response_model.model_json_schema()}\n```"

        # Determine which model to use based on model_size
        model = self.small_model if model_size == ModelSize.small else self.model
        model = model or DEFAULT_MODEL

        # Use low temperature for structured output consistency
        temperature = self.temperature if self.temperature > 0 else DEFAULT_TEMPERATURE

        response_format: dict[str, Any] = {"type": "json_object"}
        if response_model is not None:
            schema_name = getattr(response_model, "__name__", "structured_response")
            json_schema = response_model.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                },
            }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=min(temperature + 0.1 * attempt, 0.5),
                    max_tokens=max_tokens or self.max_tokens,
                    response_format=response_format,  # type: ignore[arg-type]
                )
                result = response.choices[0].message.content or ""

                # Extract JSON from response
                extracted = self._extract_json(result)

                # Retry if extraction failed (empty dict)
                if not extracted:
                    logger.warning(f"Attempt {attempt + 1}: No JSON extracted, retrying...")
                    continue

                # Validate with response_model if provided
                if response_model is not None:
                    try:
                        validated = response_model.model_validate(extracted)
                        return validated.model_dump()
                    except ValidationError as e:
                        logger.warning(f"Attempt {attempt + 1}: Validation error: {e}")
                        # Retry on validation failure
                        continue

                return extracted

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}: Error: {e}")
                continue

        # All retries exhausted
        logger.error(f"All {self.max_retries} retries failed")
        if last_error:
            raise last_error
        raise ValueError(f"Failed to generate valid response after {self.max_retries} retries")
