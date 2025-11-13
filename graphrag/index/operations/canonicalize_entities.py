# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to canonicalize the entities. We want to come out of this step with fully identified canonical entities for a given entity."""

import pandas as pd


# Current decision is to keep the raw entities and relationships and create new canonical members. The raw items will refer to this. 
# TODO SUBU: Evaluate if raw entities are useful. 
# TODO SUBU - eventually split these out into workflow steps
def canonicalize_entities(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to identify entities.

    - Identify
        - by name (__Merge_entities from before in extract_graph)
        - by system knowledge (for proper nouns, we probably have an external source of truth that can be used to identify the entity)
            - Grounding using Google / LLM is a part of this. (Ask LLM)
        - by context (like type, current doc) / time / current trends.
        - Vector similarity (should hanlde synonyms, etc.)

    - Identity Confidence.
        - 1 if solidified for sure. A score between 0 and 1 if not sure.
        - If we have enough documents referencing this entity, we can be more confident. Page Rank?
        

    - Organize Hierarchies as trees with special edges that cross RAW -> CANONICAL nodes.
        - classify the entities into types of nouns (may need context if raw data is not sure)
        - For proper nouns, create a IDENTITY HIERARCHY (various ways of identification in graph). We probably want to link to system graphs here.
        -- Proper nouns will get canonical entities per unique identity. Two guys with same name but different linkedIn profiles should be different entities.
        - For common nouns create a VARIANT HIERARCHY. Include synonyms, plurals etc.
        -- Common nouns will get a single canonical entity encompassing all variants.

    TODO 
        - system attributes for canonical entities

    """
    # prep data
    # embed entities 
    # add known attributes

    # Group by name

    # external source check

    # context check

    # Vector similarity

    return entities
