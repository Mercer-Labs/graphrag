# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pathlib import Path

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.models.language_model_config import LanguageModelConfig


class CanonicalizeEntityConfig(BaseModel):
    """Configuration section for entity canonicalization."""

    prompt: str | None = Field(
        description="The entity canonicalization prompt to use.",
        default=graphrag_config_defaults.canonicalize_entity.prompt,
    )

    max_input_tokens: int = Field(
        description="The maximum number of input tokens for the canonicalization prompt.",
        default=graphrag_config_defaults.canonicalize_entity.max_input_tokens,
    )
    strategy: dict | None = Field(
        description="Override the default entity canonicalization strategy",
        default=graphrag_config_defaults.canonicalize_entity.strategy,
    )
    model_id: str = Field(
        description="The model ID to use for entity canonicalization.",
        default=graphrag_config_defaults.canonicalize_entity.model_id,
    )

    def resolved_strategy(
        self, root_dir: str, model_config: LanguageModelConfig
    ) -> dict:
        """Get the resolved entity canonicalization strategy."""
        from graphrag.index.operations.canonicalize_entity.typing import (
            CanonicalizeStrategyType,
        )

        return self.strategy or {
            "type": CanonicalizeStrategyType.graph_intelligence,
            "llm": model_config.model_dump(),
            "canonicalization_prompt": (Path(root_dir) / self.prompt).read_text(
                encoding="utf-8"
            )
            if self.prompt
            else None,
            "max_input_tokens": self.max_input_tokens,
        }

