"""Cohere Generation Model class for flexible text generation."""

from __future__ import annotations

from typing import cast

import cohere
from pydantic import BaseModel
from typeguard import typechecked

from rago.generation.base import GenerationBase

@typechecked
class CohereGen(GenerationBase):
    """Cohere generation model for text generation."""

    default_model_name: str = 'command-r'
    default_temperature: float = 0.7
    default_output_max_length: int = 500
    default_api_params = {
        'p': 0.75,
        'k': 0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
    }

    def _setup(self) -> None:
        """Initialize the Cohere client with API key."""
        if not self.api_key:
            raise RuntimeError("API key is required for Cohere.")
        self.client = cohere.Client(self.api_key)

    def generate(self, query: str, context: list[str]) -> str | BaseModel:
        """Generate text using Cohere's API with contextual support."""
        input_text = self.prompt_template.format(
            query=query, context=' '.join(context)
        )

        if not self.client:
            raise RuntimeError('Cohere client is not initialized.')

        api_params = self.api_params if self.api_params else self.default_api_params
        model_params = {
            'model': self.model_name,
            'prompt': input_text,
            'max_tokens': self.output_max_length,
            'temperature': self.temperature,
            **api_params,
        }

        response = self.client.generate(**model_params)

        # Ensure logging structure matches test expectations
        self.logs['generation'] = {
            'model': self.model_name,
            'input_text': input_text,
            'parameters': model_params,
        }

        # Extracting generated text and ensuring it's clean
        generated_text = response.generations[0].text.strip() if response.generations else ""

        return cast(str, generated_text)
