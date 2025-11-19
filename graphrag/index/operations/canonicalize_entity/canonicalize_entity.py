# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the canonicalize_entities verb."""

from dataclasses import dataclass
from typing import Any, Tuple

import pandas as pd

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.index.operations.canonicalize_entity.typing import (
    CanonicalizationResult,
    CanonicalizationStrategy,
    CanonicalizeStrategyType,
)


# TODO SUBU: merge this with Entity in data_model.entity.py
@dataclass
class CE_EntityHolder:
    id: str
    is_raw: bool
    title: str
    attributes: list[str]
    relationships: list[str]


async def canonicalize_entity(
    r_entity: CE_EntityHolder,
    candidates: dict[Tuple[str, bool], CE_EntityHolder],
    cache: PipelineCache,
    strategy: dict[str, Any] | None = None,
) -> CanonicalizationResult:
    """Canonicalize a single entity against candidate entities, using a language model.
    
    Args:
        r_entity: The raw entity to canonicalize.
        candidates: A map of candidate entities (each as a dict with id, title, description, attributes, relationship_descriptions).
        cache: Pipeline cache for LLM responses.
        strategy: Optional strategy configuration dict.
        
    Returns:
        CanonicalizationResult with the canonical entity ID(s).
    """
    strategy = strategy or {}
    strategy_exec = load_strategy(
        strategy.get("type", CanonicalizeStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    # TODO probably can be improved - currently we are trying to send in simple id numbers instead of text, boolean combos that confuse the LLM.
    # simple id mapping
    entity_id_map: dict[str, Tuple[str, bool]] = {
        str(0): (r_entity.id, True),
        **{str(i+1): k for i, k in enumerate(candidates.keys())},
    }
    entity_id_map_inv: dict[Tuple[str, bool], str] = {v: k for k, v in entity_id_map.items()}
    entity_map = {
        str(0): r_entity,
        **{entity_id_map_inv[k]: v for k, v in candidates.items()},
    }

    candidate_map = {} # ideally a 5x5 map of candidates and their relationships
    if len(candidates) > 5:
        pass # TODO SUBU Figure out closest candidates from the ones with most similar edges rather than sending all.

    for k, ch_entity in candidates.items():
        rels = ch_entity.relationships
        if len(rels) > 5:
            pass # TODO SUBU Figure out closest relationships from raw backing edges rather than sending all.
        candidate_map[entity_id_map_inv[k]] = {
            "id": entity_id_map_inv[k],
            "title": ch_entity.title,
            "attributes": ch_entity.attributes,
            "relationship_descriptions": rels,
        }

    # Call the strategy to canonicalize the entity
    llm_result = await strategy_exec(
        "0",
        r_entity.title,
        r_entity.attributes,
        r_entity.relationships,
        candidate_map,
        cache,
        strategy_config,
    )
    if llm_result.id != "0":
        raise ValueError(f"DEVBUG: Raw entity {r_entity.id} is not the same as the chosen canonical entity {llm_result.id}.")

    return CanonicalizationResult(
        id=r_entity.id,
        canonical_entities={entity_id_map[e.id]: {
            "entity": entity_map[e.id],
            "confidence": e.confidence,
            "match_type": e.match_type,
            "reasoning": e.reasoning,
        } for e in llm_result.canonical_entities},
    )


# NOTE The strategy mechanic allows for configurably choosing different code paths.
def load_strategy(strategy_type: CanonicalizeStrategyType) -> CanonicalizationStrategy:
    """Load strategy method definition."""
    match strategy_type:
        case CanonicalizeStrategyType.graph_intelligence:
            from graphrag.index.operations.canonicalize_entity.graph_intelligence_strategy import (
                run_graph_intelligence,
            )

            return run_graph_intelligence
        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)

