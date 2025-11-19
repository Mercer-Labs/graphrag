# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CanonicalEntityExtractor' model."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from graphrag.index.operations.canonicalize_entity.typing import (
    CanonicalEntityLLMResult,
    CanonicalizationLLMResult,
    MatchType,
)
from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.canonicalize_entities import CANONICALIZE_ENTITY_PROMPT
from graphrag.tokenizer.get_tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

# this token is used in the prompt
INPUT_JSON_KEY = "input_json"


class CanonicalEntityExtractor:
    """Canonical entity extractor class definition."""

    _model: ChatModel
    _canonicalization_prompt: str
    _on_error: ErrorHandlerFn

    def __init__(
        self,
        model_invoker: ChatModel,
        max_input_tokens: int,
        canonicalization_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        self._model = model_invoker
        self._max_input_tokens = max_input_tokens
        self._tokenizer = get_tokenizer(model_invoker.config)
        self._canonicalization_prompt = canonicalization_prompt or CANONICALIZE_ENTITY_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)

    async def __call__(
        self,
        id: str,
        title: str,
        attributes: set[str] | None,
        relationship_descriptions: list[str],
        candidate_map: dict[str, dict[str, Any]],
    ) -> CanonicalizationLLMResult:
        """Call method definition."""
        # If no candidates, return current entity id as a list
        if not candidate_map:
            return CanonicalizationLLMResult(
                id=id,
                canonical_entities=[CanonicalEntityLLMResult(
                    id=id,
                    match_type=MatchType.NONE,
                    confidence=1.0,
                    reasoning="No candidates provided",
                )],
            )

        # Build the input JSON structure
        input_json = self._build_input_json(
            id=id,
            title=title,
            attributes=attributes,
            relationship_descriptions=relationship_descriptions,
            candidate_map=candidate_map,
        )

        # Format the JSON as a string for the prompt
        input_json_str = json.dumps(input_json, ensure_ascii=False, indent=4)

        prompt = self._canonicalization_prompt.format(**{INPUT_JSON_KEY: input_json_str})
        prompt_tokens = self._tokenizer.num_tokens(prompt)
        if prompt_tokens > self._max_input_tokens:
            raise ValueError(f"Canonicalization Prompt is too long: {prompt_tokens} tokens > {self._max_input_tokens} tokens")

        # Call LLM to find the best match
        response = await self._model.achat(
            prompt,
            name="canonicalize_entity",
            json=True,
            json_model=CanonicalizationLLMResult,
        )

        # Extract canonical entity IDs from parsed response
        if response.parsed_response:
            return response.parsed_response
        # Fallback if parsing failed
        # TODO SUBU This needs to be a better error handling / trigger reprocessing pipeline
        logger.warning(
            f"Failed to parse response as Pydantic model, using current entity ID: {id}"
        )
        return CanonicalizationLLMResult(
            id=id,
            canonical_entities=[CanonicalEntityLLMResult(
                id=id,
                match_type=MatchType.NONE,
                confidence=1.0,
                reasoning="Failed to parse response as Pydantic model",
            )],
        )

    def _build_input_json(
        self,
        id: str,
        title: str,
        attributes: set[str] | None,
        relationship_descriptions: list[str],
        candidate_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the input JSON structure for the prompt."""
        # Build entity structure
        entity = {
            "id": id,
            "title": title,
            "attributes": list(attributes or []),
            "relationship_descriptions": relationship_descriptions or [],
        }

        # Build candidates structure as a dict keyed by candidate ID
        return {"entity": entity, "candidates": candidate_map}

