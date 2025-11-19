# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_graph_intelligence method to run graph intelligence for entity canonicalization."""

import logging
from typing import Any

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.index.operations.canonicalize_entity.canonical_entity_extractor import (
    CanonicalEntityExtractor,
    CanonicalizationLLMResult,
)
from graphrag.index.operations.canonicalize_entity.typing import StrategyConfig
from graphrag.language_model.manager import ModelManager
from graphrag.language_model.protocol.base import ChatModel

logger = logging.getLogger(__name__)


async def run_graph_intelligence(
    id: str,
    title: str,
    attributes: set[str] | None,
    relationship_descriptions: list[str],
    candidate_map: dict[str, dict[str, Any]],
    cache: PipelineCache,
    args: StrategyConfig,
) -> CanonicalizationLLMResult:
    """Run the graph intelligence entity canonicalization strategy."""
    llm_config = LanguageModelConfig(**args["llm"])
    max_input_tokens = args["max_input_tokens"]
    llm = ModelManager().get_or_create_chat_model(
        name="canonicalize_entities",
        model_type=llm_config.type,
        config=llm_config,
        cache=cache,
    )

    return await run_canonicalize_entity(
        llm,
        id,
        title,
        attributes,
        relationship_descriptions,
        candidate_map,
        max_input_tokens,
        args,
    )


async def run_canonicalize_entity(
    model: ChatModel,
    id: str,
    title: str, 
    attributes: set[str] | None,
    relationship_descriptions: list[str],
    candidate_map: dict[str, dict[str, Any]],
    max_input_tokens: int,
    args: StrategyConfig,
) -> CanonicalizationLLMResult:
    """Run the entity canonicalization chain."""
    canonicalization_prompt = args.get("canonicalization_prompt", None)
    extractor = CanonicalEntityExtractor(
        model_invoker=model,
        canonicalization_prompt=canonicalization_prompt,
        max_input_tokens=max_input_tokens,
        on_error=lambda e, stack, details: logger.error(
            "Entity Canonicalization Error",
            exc_info=e,
            extra={"stack": stack, "details": details},
        ),
    )

    return await extractor(
        id=id,
        title=title,
        attributes=attributes,
        relationship_descriptions=relationship_descriptions,
        candidate_map=candidate_map,
    )

