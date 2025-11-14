# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
from typing import Any

import pandas as pd

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.embeddings import (
    raw_entity_title_embedding,
    raw_relationship_description_embedding,
)
from graphrag.config.enums import AsyncType
from graphrag.config.get_embedding_settings import get_embedding_settings
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.canonicalize_entities import canonicalize_entities
from graphrag.index.operations.canonicalize_relationships import (
    canonicalize_relationships,
)
from graphrag.index.operations.embed_text.embed_text import embed_text
from graphrag.index.operations.extract_graph.extract_graph import (
    extract_graph as extractor,
)
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    summarize_descriptions,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to create the base entity graph."""
    logger.info("Workflow started: extract_graph")
    text_units = await load_table_from_storage("text_units", context.output_storage)

    extract_graph_llm_settings = config.get_language_model_config(
        config.extract_graph.model_id
    )
    extraction_strategy = config.extract_graph.resolved_strategy(
        config.root_dir, extract_graph_llm_settings
    )

    summarization_llm_settings = config.get_language_model_config(
        config.summarize_descriptions.model_id
    )
    summarization_strategy = config.summarize_descriptions.resolved_strategy(
        config.root_dir, summarization_llm_settings
    )

    raw_entities, raw_relationships = await extract_raw_graph(
        text_units=text_units,
        callbacks=context.callbacks,
        cache=context.cache,
        extraction_strategy=extraction_strategy,
        extraction_num_threads=extract_graph_llm_settings.concurrent_requests,
        extraction_async_mode=extract_graph_llm_settings.async_mode,
        document_type=config.extract_graph.document_type,
    )

    text_embed_config_strategy = get_embedding_settings(config)["strategy"]

    raw_entities, raw_relationships = await preprocess_raw_graph(
        raw_entities=raw_entities,
        raw_relationships=raw_relationships,
        callbacks=context.callbacks,
        cache=context.cache,
        text_embed_config_strategy=text_embed_config_strategy,
    )

    if config.snapshots.raw_graph:
        await write_table_to_storage(raw_entities.drop(columns=["embedding"]), "raw_entities", context.output_storage)
        await write_table_to_storage(raw_relationships.drop(columns=["embedding"]), "raw_relationships", context.output_storage)

    entities, relationships = await process_raw_graph(
        raw_entities=raw_entities,
        raw_relationships=raw_relationships,
        callbacks=context.callbacks,
        cache=context.cache,
        summarization_strategy=summarization_strategy,
        summarization_num_threads=summarization_llm_settings.concurrent_requests,
        text_embed_config_strategy=text_embed_config_strategy,
    )

    await write_table_to_storage(entities, "entities", context.output_storage)
    await write_table_to_storage(relationships, "relationships", context.output_storage)

    logger.info("Workflow completed: extract_graph")
    return WorkflowFunctionOutput(
        result={
            "entities": entities,
            "relationships": relationships,
        }
    )


async def extract_raw_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    extraction_strategy: dict[str, Any] | None = None,
    extraction_num_threads: int = 4,
    extraction_async_mode: AsyncType = AsyncType.AsyncIO,
    document_type: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All the steps to create the raw entity graph."""
    # this returns a graph for each text unit, to be merged later
    extracted_entities, extracted_relationships = await extractor(
        text_units=text_units,
        callbacks=callbacks,
        cache=cache,
        text_column="text",
        id_column="id",
        strategy=extraction_strategy,
        async_mode=extraction_async_mode,
        document_type=document_type,
        num_threads=extraction_num_threads,
    )

    if not _validate_data(extracted_entities):
        error_msg = "Entity Extraction failed. No entities detected during extraction."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not _validate_data(extracted_relationships):
        error_msg = (
            "Entity Extraction failed. No relationships detected during extraction."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    return (extracted_entities, extracted_relationships)

async def preprocess_raw_graph(
    raw_entities: pd.DataFrame,
    raw_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    text_embed_config_strategy: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All the steps to preprocess the raw graph."""

    #Embed raw entities and relationships
    raw_entities["embedding"] = await embed_text(
        input=raw_entities.loc[:, ["id", "title"]],
        callbacks=callbacks,
        cache=cache,
        embedding_name=raw_entity_title_embedding,
        embed_column="title",
        strategy=text_embed_config_strategy,
    )
    raw_relationships["embedding"] = await embed_text(
        input=raw_relationships.loc[:, ["key", "text_description"]],
        callbacks=callbacks,
        cache=cache,
        embedding_name=raw_relationship_description_embedding,
        id_column="key",
        embed_column="text_description",
        strategy=text_embed_config_strategy,
    )
    return (raw_entities, raw_relationships)

async def process_raw_graph(
    raw_entities: pd.DataFrame,
    raw_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    text_embed_config_strategy: dict,
    summarization_strategy: dict[str, Any] | None = None,
    summarization_num_threads: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process the raw graph."""
    canonical_entities, canonical_relationships = await canonicalize_graph(
        entities=raw_entities,
        relationships=raw_relationships,
        text_embed_config_strategy=text_embed_config_strategy,
    )
    
    entities, relationships = await get_summarized_entities_relationships(
        extracted_entities=canonical_entities,
        extracted_relationships=canonical_relationships,
        callbacks=callbacks,
        cache=cache,
        summarization_strategy=summarization_strategy,
        summarization_num_threads=summarization_num_threads,
    )

    return (entities, relationships)

async def canonicalize_graph(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    text_embed_config_strategy: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Canonicalize the graph.
    """
    canonical_entities = canonicalize_entities(entities, relationships, pd.DataFrame(), pd.DataFrame(), text_embed_config_strategy)
    canonical_relationships = canonicalize_relationships(relationships)
    return (canonical_entities, canonical_relationships)


async def get_summarized_entities_relationships(
    extracted_entities: pd.DataFrame,
    extracted_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    summarization_strategy: dict[str, Any] | None = None,
    summarization_num_threads: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the entities and relationships."""
    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=callbacks,
        cache=cache,
        strategy=summarization_strategy,
        num_threads=summarization_num_threads,
    )

    # TODO SUBU - There are duplicate titles and source/targets: This causes a cartesian product explosion ... fix it.
    relationships = extracted_relationships.drop(columns=["description"]).merge(
        relationship_summaries, on=["source", "target"], how="left"
    )

    #extracted_entities.drop(columns=["attributes"], inplace=True)
    entities = extracted_entities.merge(entity_summaries, on="title", how="left")
    return entities, relationships


def _validate_data(df: pd.DataFrame) -> bool:
    """Validate that the dataframe has data."""
    return len(df) > 0
