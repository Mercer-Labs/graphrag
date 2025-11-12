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
from uuid import uuid1

import networkx as nx
from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
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
    source_docs: dict[Any, Any]


class RawEntityModel(BaseModel):
    """A model for an entity."""

    entity_id: str = Field(description="The id of the entity.")
    entity_name: str = Field(description="The name of the entity.")
    entity_type: str = Field(description="The type of the entity.")
    entity_attributes: list[str] = Field(description="The attributes of the entity.")


class RawRelationshipModel(BaseModel):
    """A model for a relationship."""
    source_entity_id: str = Field(description="The id of the source entity.")
    target_entity_id: str = Field(description="The id of the target entity.")
    relationship_description: str = Field(description="The description of the relationship.")
    relationship_strength: float = Field(description="The strength of the relationship.")


class ExtractGraphResponse(BaseModel):
    """A model for the expected LLM response shape."""

    entities: list[RawEntityModel] = Field(description="A list of entities identified.")
    relationships: list[RawRelationshipModel] = Field(description="A list of relationships identified.")


class GraphExtractor:
    """Unipartite graph extractor class definition."""

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
        self, texts: list[str], prompt_variables: dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        """Call method definition."""
        if prompt_variables is None:
            prompt_variables = {}
        all_records: dict[int, list[ExtractGraphResponse]] = {}
        source_doc_map: dict[int, str] = {}

        # Wire defaults into the prompt variables
        prompt_variables = {
            **prompt_variables,
            self._document_type_key: prompt_variables.get(self._document_type_key)
            or DEFAULT_DOCUMENT_TYPE,
        }

        for doc_index, text in enumerate(texts):
            try:
                # Invoke the entity extraction
                result = await self._process_document(text, prompt_variables)
                source_doc_map[doc_index] = text
                all_records[doc_index] = result
            except Exception as e:
                logger.exception("error extracting graph")
                self._on_error(
                    e,
                    traceback.format_exc(),
                    {
                        "doc_index": doc_index,
                        "text": text,
                    },
                )

        output = await self._process_results(all_records)

        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
        )

    async def _process_document(
        self, text: str, prompt_variables: dict[str, str]
    ) -> list[ExtractGraphResponse]:
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
        results: dict[int, list[ExtractGraphResponse]],
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
        for source_doc_id, extracted_data in results.items():
            source_doc_id_str = str(source_doc_id)
            entity_key_map: dict[str, str] = {}
            for response in extracted_data:
                for entity in response.entities:
                    # names are used as titles from here.
                    entity_title = clean_str(entity.entity_name)
                    entity_type = clean_str(entity.entity_type)
                    entity_id = clean_str(entity.entity_id)
                    entity_attributes = set([clean_str(attribute) for attribute in entity.entity_attributes])
                    
                    if entity_id in entity_key_map:
                        merge = True
                    else:
                        merge = False
                        entity_key_map[entity_id] = uuid1().hex
                    unique_entity_id = entity_key_map[entity_id]
                    if merge:
                        node = graph.nodes[unique_entity_id]
                        # We don't expect title/type mismatches, but it's possible because LLMs. We just pick the longer one for now.
                        # TODO SUBU - figure out a reprocessing pipeline.
                        if node["title"] != entity_title:
                            logger.warning(f"Entity title mismatch: {node['title']} != {entity_title} for entity {entity_id}")
                            node["title"] = entity_title if len(entity_title) > len(node["title"]) else node["title"]
                        if node["type"] != entity_type:
                            logger.warning(f"Entity type mismatch: {node['type']} != {entity_type} for entity {entity_id}")
                            node["type"] = entity_type if len(entity_type) > len(node["type"]) else node["type"]
                        node["attributes"].update(entity_attributes)
                        node["source_id"].add(source_doc_id)
                    else:
                        graph.add_node(
                            unique_entity_id,
                            title=entity_title,
                            type=entity_type,
                            attributes=entity_attributes,
                            source_id={source_doc_id},
                        )
                for relationship in response.relationships:
                    if relationship.source_entity_id not in entity_key_map or relationship.target_entity_id not in entity_key_map:
                        # TODO SUBU - reprocessing pipeline: handle missing edge links better
                        logger.warning(f"Error processing document id: {source_doc_id}: Source or target entity not found: {relationship.source_entity_id} \
                            or {relationship.target_entity_id}. Skipping relationship {relationship}")
                        continue
                    source_entity_id = entity_key_map[relationship.source_entity_id]
                    target_entity_id = entity_key_map[relationship.target_entity_id]

                    # TODO SUBU see if we should move to relationship attributes instead of description.
                    relationship_description = clean_str(relationship.relationship_description)
                    relationship_strength = relationship.relationship_strength
                    graph.add_edge(
                        source_entity_id,
                        target_entity_id,
                        strength=relationship_strength,
                        weight=1.0,
                        description=relationship_description,
                        source_id=source_doc_id,
                    )

        return graph

