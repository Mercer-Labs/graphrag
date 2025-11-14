# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'GraphExtractionResult' and 'GraphExtractor' models."""

import hashlib
import json
import logging
import re
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field
from uuid_utils import uuid7

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.index.operations.extract_graph.node_references import NodeReferences
from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.index.utils.string import clean_str
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.extract_graph import (
    CONTINUE_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    LOOP_PROMPT,
)

DEFAULT_DOCUMENT_TYPE = "DOCUMENT"

logger = logging.getLogger(__name__)


@dataclass
class GraphExtractionResult:
    """Multigraph extraction result class definition."""

    output: nx.MultiGraph
    source_doc: str


class RawEntityModel(BaseModel):
    """A model for an entity."""

    id: str = Field(description="The id of the entity.")
    name: str = Field(description="The name of the entity.")
    type: str = Field(description="The type of the entity.")
    attributes: list[str] = Field(description="The attributes of the entity.")
    is_proper_noun: bool = Field(description="Whether the entity is a proper noun.")


class RawRelationshipModel(BaseModel):
    """A model for a relationship."""
    source: str = Field(description="The id of the source entity.")
    target: str = Field(description="The id of the target entity.")
    text_location: int = Field(description="The start location index of the relationship in the provided text")
    description: str = Field(description="The description of the relationship.")
    strength: float = Field(description="The strength of the relationship.")


class ExtractGraphResponse(BaseModel):
    """A model for the expected LLM response shape."""

    entities: list[RawEntityModel] = Field(description="A list of entities identified.")
    relationships: list[RawRelationshipModel] = Field(description="A list of relationships identified.")


class GraphExtractor:
    """Multipartite graph extractor class definition."""
    _model: ChatModel
    _document_type_key: str
    _input_text_key: str
    _entity_name_key: str
    _extraction_prompt: str
    _max_gleanings: int
    _on_error: ErrorHandlerFn

    def __init__(
        self,
        model_invoker: ChatModel,
        input_text_key: str | None = None,
        document_type_key: str | None = None,
        prompt: str | None = None,
        max_gleanings: int | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        # TODO: streamline construction
        self._model = model_invoker
        self._input_text_key = input_text_key or "input_text"
        self._document_type_key = document_type_key or "document_type"
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else graphrag_config_defaults.extract_graph.max_gleanings
        )
        self._on_error = on_error or (lambda _e, _s, _d: None)

    async def __call__(
        self, text: str, prompt_variables: dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        """Call method definition."""
        if prompt_variables is None:
            prompt_variables = {}

        # Wire defaults into the prompt variables
        prompt_variables = {
            **prompt_variables,
            self._document_type_key: prompt_variables.get(self._document_type_key)
            or DEFAULT_DOCUMENT_TYPE,
        }

        results: list[ExtractGraphResponse] = []
        try:
            # Invoke the entity extraction
            results = await self._process_document(text, prompt_variables)
        except Exception as e:
            logger.exception("error extracting graph")
            self._on_error(
                e,
                traceback.format_exc(),
                {
                    "text": text,
                },
            )

        output = await self._process_results(results, text)

        return GraphExtractionResult(
            output=output,
            source_doc=text,
        )

    async def _process_document(
        self, text: str, prompt_variables: dict[str, str]
    ) -> list[ExtractGraphResponse]:
        # Simple protection against usage of Placeholders: For now this should be rare enough.
        # TODO SUBU make this better.
        text = NodeReferences.cleanup_placeholders_in_text(text)
        response = await self._model.achat(
            self._extraction_prompt.format(**{
                **prompt_variables,
                self._input_text_key: text,
            }),
            json=True,
            name="extract_graph",
            json_model=ExtractGraphResponse,
        )
        results = [response.parsed_response or ExtractGraphResponse(
            entities=[],
            relationships=[],
        )]

        # if gleanings are specified, enter a loop to extract more entities
        # there are two exit criteria: (a) we hit the configured max, (b) the model says there are no more entities
        if self._max_gleanings > 0:
            for i in range(self._max_gleanings):
                response = await self._model.achat(
                    LOOP_PROMPT,
                    name=f"extract_graph_loopcheck-{i}",
                    history=response.history,
                )
                if response.output.content.lower() != "y":
                    break

                response = await self._model.achat(
                    CONTINUE_PROMPT,
                    name=f"extract_graph_continuation-{i}",
                    history=response.history,
                    json=True,
                    json_model=ExtractGraphResponse,
                )
                results.append(response.parsed_response or ExtractGraphResponse(
                    entities=[],
                    relationships=[],
                ))

        return results

    async def _process_results(
        self,
        results: list[ExtractGraphResponse],
        text: str,
    ) -> nx.MultiGraph:
        """Parse the results from each doc: Can contain multiple Responses from retries etc.

        Args:
            - results - dict of results from the extraction chain: Each result is a json object with the following fields:
                - entities: list of entities 
                - relationships: list of relationships 
        Returns:
            - output - unipartite graph in graphML format
        """       
        graph = nx.MultiGraph()
        entity_key_map: dict[str, str] = {}
        for response in results:
            for entity in response.entities:
                # names are used as titles from here.
                entity_title = clean_str(entity.name)
                entity_type = clean_str(entity.type)
                entity_id = clean_str(entity.id)
                entity_attributes = set([clean_str(attribute) for attribute in entity.attributes])
                
                if entity_id in entity_key_map:
                    merge = True
                else:
                    merge = False
                    entity_key_map[entity_id] = uuid7().hex # use this for time-sortability
                unique_entity_id = entity_key_map[entity_id]
                if merge:
                    node = graph.nodes[unique_entity_id]
                    # We don't expect title/type mismatches, but it's possible because LLMs. We just pick the longer one for now.
                    # TODO SUBU - figure out a reprocessing pipeline.
                    if node["title"] != entity_title:
                        logger.warning(f"Entity title mismatch: {node['title']} != {entity_title} for entity {entity_id}")
                        node["title"] = entity_title if len(entity_title) > len(node["title"]) else node["title"]
                    if node["llm_inferred_type"] != entity_type:
                        logger.warning(f"Entity type mismatch: {node['llm_inferred_type']} != {entity_type} for entity {entity_id}")
                        node["llm_inferred_type"] = entity_type if len(entity_type) > len(node["llm_inferred_type"]) else node["llm_inferred_type"]
                    node["attributes"].update(entity_attributes)
                else:
                    graph.add_node(
                        unique_entity_id,
                        title=entity_title,
                        llm_inferred_type=entity_type,
                        attributes=entity_attributes,
                        is_proper_noun=entity.is_proper_noun,
                    )
            for relationship in response.relationships:
                if relationship.source not in entity_key_map or relationship.target not in entity_key_map:
                    # TODO SUBU - reprocessing pipeline: handle missing edge links better
                    logger.warning(f"Error processing document text: {text}: Source or target entity not found: {relationship.source} \
                        or {relationship.target}. Skipping relationship {relationship}")
                    continue
                source_entity_id = entity_key_map[relationship.source]
                target_entity_id = entity_key_map[relationship.target]

                # TODO SUBU see if we should move to relationship attributes instead of description.
                cleaned_desc = NodeReferences.encode_node_references_in_llm_output(clean_str(relationship.description), source_entity_id, target_entity_id)
                text_description = NodeReferences.hydrate_node_references(cleaned_desc, graph.nodes)
                graph.add_edge(
                    source_entity_id,
                    target_entity_id,
                    key=uuid7().hex,
                    strength=relationship.strength,
                    weight=1.0,
                    description=cleaned_desc,
                    text_description=text_description,
                    text_location=relationship.text_location,
                )

        return graph

