# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CanonicalizationResult' model."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

from graphrag.cache.pipeline_cache import PipelineCache

StrategyConfig = dict[str, Any]

from pydantic import BaseModel, Field


class MatchType(str, Enum):
    """MatchType class definition."""

    EXACT = "exact"
    PARTIAL = "partial"
    NONE = "none"

class CanonicalEntityLLMResult(BaseModel):
    id: str = Field(..., description="The ID of the canonical entity.")
    match_type: MatchType = Field(..., description="The type of match between the current entity and the canonical entity.")
    confidence: float = Field(..., description="The confidence of the match between the current entity and the canonical entity, between 0 and 1.")
    reasoning: str = Field(..., description="The reasoning for why it is a good match.")

class CanonicalizationLLMResult(BaseModel):
    id: str = Field(..., description="The ID of the current entity being canonicalized.")
    canonical_entities: list[CanonicalEntityLLMResult] = Field(..., description="The map of canonical entity IDs to LLM's result about why it is a good match")


@dataclass
class CanonicalizationResult:
    """Entity canonicalization result class definition."""

    id: str
    """The ID of the current entity being canonicalized."""

    canonical_entities: dict[Tuple[str, bool], dict[str, Any]]
    """The map of canonical entity IDs to entity, LLM's confidence and reasoning about why it is a good match."""


CanonicalizationStrategy = Callable[
    [
        str,  # current_entity_id
        str,  # current_entity_title
        str,  # current_entity_type
        set[str] | None,  # current_entity_attributes
        list[str],  # relationship_descriptions
        dict[str, dict[str, Any]],  # candidate_map (map of candidate ID to dict with id, title, attributes, relationship_descriptions)
        PipelineCache,
        StrategyConfig,
    ],
    Awaitable[CanonicalizationLLMResult],
]


class CanonicalizeStrategyType(str, Enum):
    """CanonicalizeStrategyType class definition."""

    graph_intelligence = "graph_intelligence"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'

