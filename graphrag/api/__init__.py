# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""API for GraphRAG.

WARNING: This API is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from graphrag.api.index import build_index
from graphrag.api.query import (
    basic_search,
    basic_search_streaming,
    drift_search,
    drift_search_streaming,
    global_search,
    global_search_streaming,
    local_search,
    local_search_streaming,
    multi_index_basic_search,
    multi_index_drift_search,
    multi_index_global_search,
    multi_index_local_search,
)

__all__ = [  # noqa: RUF022
    # index API
    "build_index",
    # query API
    "global_search",
    "global_search_streaming",
    "local_search",
    "local_search_streaming",
    "drift_search",
    "drift_search_streaming",
    "basic_search",
    "basic_search_streaming",
    "multi_index_basic_search",
    "multi_index_drift_search",
    "multi_index_global_search",
    "multi_index_local_search",
]
