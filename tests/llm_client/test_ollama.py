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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.llm_client import OllamaClient, OllamaConfig
from graphiti_core.prompts.models import Message


class TestOllamaConfig:
    def test_default_values(self):
        config = OllamaConfig()
        assert config.model == 'llama3'
        assert config.base_url == 'http://localhost:11434/v1'
        assert config.api_key == 'ollama'
        assert config.max_tokens == 2048

    def test_custom_values(self):
        config = OllamaConfig(
            model='codellama',
            base_url='http://custom:11434/v1',
            api_key='custom-key',
            max_tokens=4096,
        )
        assert config.model == 'codellama'
        assert config.base_url == 'http://custom:11434/v1'
        assert config.api_key == 'custom-key'
        assert config.max_tokens == 4096


class TestOllamaClient:
    @pytest.fixture
    def mock_openai_client(self):
        with patch('graphiti_core.llm_client.ollama_client.AsyncOpenAI') as mock:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=MagicMock(
                    choices=[MagicMock(message=MagicMock(content='{"result": "test"}'))]
                )
            )
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_generate_response(self, mock_openai_client):
        client = OllamaClient()
        messages = [Message(role='user', content='Hello')]

        result = await client._generate_response(messages)

        assert result == {'result': 'test'}
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_with_custom_model(self, mock_openai_client):
        config = OllamaConfig(model='mistral')
        client = OllamaClient(config)
        messages = [Message(role='user', content='Hello')]

        await client._generate_response(messages)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == 'mistral'

    @pytest.mark.asyncio
    async def test_generate_response_with_small_model(self, mock_openai_client):
        config = OllamaConfig(model='llama3', small_model='llama3:7b')
        client = OllamaClient(config)
        from graphiti_core.llm_client.config import ModelSize

        messages = [Message(role='user', content='Hello')]

        await client._generate_response(messages, model_size=ModelSize.small)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == 'llama3:7b'
