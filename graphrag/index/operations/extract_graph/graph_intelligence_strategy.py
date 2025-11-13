# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_graph_intelligence,  run_extract_graph and _create_text_splitter methods to run graph intelligence."""

import logging

import networkx as nx

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.index.operations.extract_graph.typing import (
    Document,
    EntityExtractionResult,
    StrategyConfig,
)
from graphrag.data_model.schemas import SystemAttributes
from graphrag.language_model.manager import ModelManager
from graphrag.language_model.protocol.base import ChatModel

logger = logging.getLogger(__name__)

async def run_graph_intelligence(
    doc: Document,
    document_type: str,
    cache: PipelineCache,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the graph intelligence entity extraction strategy."""
    llm_config = LanguageModelConfig(**args["llm"])

    llm = ModelManager().get_or_create_chat_model(
        name="extract_graph",
        model_type=llm_config.type,
        config=llm_config,
        cache=cache,
    )

    return await run_extract_graph(llm, doc, document_type, args)


async def run_extract_graph(
    model: ChatModel,
    doc: Document,
    document_type: str,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the entity extraction chain."""
    extraction_prompt = args.get("extraction_prompt", None)
    max_gleanings = args.get(
        "max_gleanings", graphrag_config_defaults.extract_graph.max_gleanings
    )

    extractor = GraphExtractor(
        model_invoker=model,
        prompt=extraction_prompt,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: logger.error(
            "Entity Extraction Error", exc_info=e, extra={"stack": s, "details": d}
        ),
    )
    text = doc.text.strip()

    results = await extractor(
        text,
        {
            "document_type": document_type,
        },
    )

    graph = results.output
    # Map the "source_id" back to the "id" field
    for _, node in graph.nodes(data=True):  # type: ignore
        if node is not None:
            node["source_id"] = doc.id
            node[SystemAttributes.RAW] = True

    for _, _, edge in graph.edges(data=True):  # type: ignore
        if edge is not None:
            edge["source_id"] = doc.id
            edge[SystemAttributes.RAW] = True

    entities = [
        ({"id": item[0], **(item[1] or {})})
        for item in graph.nodes(data=True)
        if item is not None
    ]

    relationships = nx.to_pandas_edgelist(graph)

    return EntityExtractionResult(entities, relationships, graph)
