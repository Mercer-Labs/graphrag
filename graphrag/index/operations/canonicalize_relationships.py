# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to canonicalize the relationships. We want to come out of this step with fully identified canonical relationships for a given relationship."""

import pandas as pd


async def canonicalize_relationships(
    raw_relationships: pd.DataFrame,
) -> pd.DataFrame:
    """
    The raw entities are related to each other via LLM Determined edges. We have created (or mapped) canonical entities for the raw entities that grounds
    them to a particular real-world entity. Here we canonicalize the relationships between the canonical entities.
    - we try to summarize the existing relationships between the canonical entities into one.
    - We try to assign system edges if any. 
    - TODO SUBU Think about normalization(building known edge types) vs vector similarity matching (the current plan).
    """

    return raw_relationships
