# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to canonicalize the entities. We want to come out of this step with fully identified canonical entities for a given entity."""

import logging
from typing import Any, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from uuid_utils import uuid7

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.config.embeddings import (
    canonical_entity_title_embedding,
    raw_entity_title_embedding,
)
from graphrag.data_model.schemas import RelationshipType, SystemAttributes
from graphrag.index.operations.canonicalize_entity.canonicalize_entity import (
    CE_EntityHolder,
    canonicalize_entity,
)
from graphrag.index.operations.canonicalize_entity.typing import (
    CanonicalizationResult,
    MatchType,
)
from graphrag.index.operations.embed_text.embed_text import get_vector_store_for_write
from graphrag.utils.api import get_vector_store_for_query
from graphrag.vector_stores.base import BaseVectorStore, VectorStoreDocument

logger = logging.getLogger(__name__)

# slightly generous: around .15 seems to include plurals etc. But leaving it here for better consolidation with more context.
VECTOR_SEARCH_DISTANCE_THRESHOLD = 0.2


def search_with_threshold(embedding, vector_store: BaseVectorStore, k: int, threshold: float = VECTOR_SEARCH_DISTANCE_THRESHOLD):
    return [result for result in vector_store.similarity_search_by_vector(embedding, k=k) if result.score < threshold]



# Current decision is to keep the raw entities and relationships and create new canonical members. The raw items will refer to this. 
# TODO SUBU: Evaluate if raw entities are useful. 
# TODO SUBU - eventually split these out into workflow steps
"""
The knowledge graph consists of:
- System data (from system knowledge)
Today we have a system Identity hierarchy with linkedIn profile info. (name, profile url, etc.). Possibly Twitter.
For Fake company we will have AD like system where we have user identity info, his google/slack/github etc account information. We will have inter relationships between these identities like teams etc. We add nodes and system relationships as we discover more info.
- Inferred data (from LLM)
This processing system generates raw and canonical data that get stored here as well. These refer to the system data and vice-versa.
"""
async def canonicalize_entities(
    raw_entities: pd.DataFrame,
    raw_relationships: pd.DataFrame,
    known_identities: pd.DataFrame,
    known_relationships: pd.DataFrame,
    text_embed_config_strategy: dict,
    canonicalization_strategy: dict,
    cache: PipelineCache,
    num_threads: int = 4,
    canonical_entities: pd.DataFrame | None = None, # "id", "title", "title_SS_embedding", "title_RD_embedding", "metadata", "summary", "summary_RD_embedding", "raw_entity_ids"
    canonical_relationships: pd.DataFrame | None = None, #"id", "source", "target", "metadata", "canonical_summary", "canonical_summary_RD_embedding"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """All the steps to identify entities.

    - Identify
        - Vector similarity (should handle name similarities, synonyms, etc.) to existing canonical entities.
        TODO Eventually improve this: See resolve_extracted_nodes in graphiti_core/utils/maintenance/node_operations.py
        -
        - by system knowledge (for proper nouns, we probably have an external source of truth that can be used to identify the entity). Using attributes and vectors
        -
        - by context (like type, current doc) / time / current trends. (We probably need vector search + summary)
        - Grounding using Google / LLM. (Ask LLM)
        
    

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
        - should we cluster raw entities by k-means (scikit)?
        - graph maintenance jobs that continuously evaluate canonicalization / hierarchies etc.
        - evaluate pandarell - generally parallelize. This lib has asyncIO only for LLM stuff.
    """

    # k-means cluster the raw entities. 
    # We want to organize by title similarity (graph partitioning doesn't work because raws probably have duplicates and island clusters)
    # TODO SUBU we assume there are num_threads clusters. In reality there may not be: Look at silhouette score/analysis to pick this number better.
    kmeans = KMeans(n_clusters=num_threads, random_state=42)
    k_clusters = kmeans.fit(raw_entities["title_SS_embedding"].tolist()) # Probably a better way to do this...
    raw_entities["cluster"] = k_clusters.labels_
    raw_entity_cluster_dfs = {cluster: group for cluster, group in raw_entities.groupby("cluster")}

    # Temporarily we reconstruct the entire graph and manipulate the same - editing a dataframe is nuts. Eventually this should be a store.
    r_graph = nx.from_pandas_edgelist(
        raw_relationships,
        source="source", 
        target="target", 
        edge_attr=["description", "edge_type", "weight", "strength", "text_description", "text_description_SS_embedding", "text_location", "source_id", "human_readable_id"],
        edge_key="key",
        create_using=nx.MultiGraph,
    )
    for _, row in raw_entities.iterrows():
        r_graph.add_node(
            row["id"],
            title=row["title"],
            title_SS_embedding=row["title_SS_embedding"],
            is_proper_noun=row["is_proper_noun"],
            source_id=row["source_id"],
            llm_inferred_type=row["llm_inferred_type"],
            node_type=row["node_type"],
            attributes=row["attributes"],
            human_readable_id=row["human_readable_id"],
        )
    
    if canonical_relationships is None:
        c_graph = nx.Graph()
    else:
        c_graph = nx.from_pandas_edgelist(canonical_relationships, source="source", target="target", edge_attr=["metadata", "canonical_summary", "canonical_summary_RD_embedding"])
    
    if canonical_entities is not None:
        for _, row in canonical_entities.iterrows():
            c_graph.add_node(
                row["id"],
                title=row["title"],
                title_SS_embedding=row["title_SS_embedding"],
                metadata=row["metadata"],
                summary=row["summary"],
                # TODO SUBU see if we need to keep a known identity list (if anything matches, it's a match)
                # TODO SUBU see if we need to keep a disambiguation list (ways to differentiate between entities)
                # TODO SUBU see if we need a negative list (entities / attributes that are not candidates)
                summary_RD_embedding=row["summary_RD_embedding"],
                raw_entity_ids=row["raw_entity_ids"],
            )

    logger.debug(f"Vector Store Args: {text_embed_config_strategy['vector_store']}")
    c_entity_title_vs = get_vector_store_for_write(
        vector_store_config=text_embed_config_strategy['vector_store'],
        llm_config=text_embed_config_strategy['llm'],
        embedding_name=canonical_entity_title_embedding,
    )

    # NOTE Because of the parallelization, we cannot avoid duplicate canonical entities getting created. So we need a later compaction round to consolidate 
    # them. The compaction needs to be done with context and probably graph shape analysis AND offline. Once this is cross machine, the simplistic finalization
    # here won't be enough. To reduce the duplicates as much as possible, we do serialize within each thread here. Compaction can probably also be triggered by
    # queries etc.
    # 
    # NOTE: This merging is what makes the representative raw-entity embeddings very useful for future canonicalization. It tries to bring the LLM intel to 
    # embedding vectors for name searches.
    #
    # TODO SUBU PARALLELIZE the below.
    #  we maintain a list of the prospects instead of directly creating canonical entities here. Because without canonical edges, these are
    #  not useful for context checks - will only confuse the LLM. We stash the entity holders for reuse.
    new_c_entity_prospects: dict[str, CE_EntityHolder] = {}
    raw_entity_map = {} # map of raw_id -> raw_entity
    raw_entity_to_canonical_entity_map = {} # map of raw_id -> set(canonical_entity_ids)
    for cluster, cluster_df in raw_entity_cluster_dfs.items():
        logger.debug(f"Processing cluster {cluster} with {len(cluster_df)} raw entities.")
        # nx is not threadsafe.
        # TODO SUBU - remember this is NOT a deep copy / foolproof. You can still edit the nodes here...
        c_graph_view = nx.subgraph_view(c_graph) 
        for _, row in cluster_df.iterrows():
            # -- start thread
            logger.debug(f"Processing raw entity {row['id']} in cluster {cluster}.")
            # We create the vector store one per thread because we want to set query filters for each thread.
            r_entity_title_vs = get_vector_store_for_query(
                vector_store_args={
                    text_embed_config_strategy['embed_text_vector_store_id']: text_embed_config_strategy['vector_store'],
                },
                llm_config=text_embed_config_strategy['llm'],
                embedding_name=raw_entity_title_embedding,
            )
            raw_entity_node = r_graph.nodes[row["id"]]
            # TODO SUBU: Add the system relationships from documents / authors etc to help infer better? We can dig lot more using those 'known' 
            # relationships. (Expand DFS to other nodes they have created that are like this one...)
            raw_entity_holder = raw_entity_map[row["id"]] = CE_EntityHolder(
                    id=row["id"],
                    is_raw=True,
                    title=row["title"],
                    attributes=raw_entity_node["attributes"],
                    relationships=[edge[2]["text_description"] for edge in r_graph.edges(row["id"], data=True)],
                )
            # find candidate canonical entities by vector search on canonical entities.
            # NOTE this canonical DF gets updated as we go along. This will probably require special handling when we parallelize 
            candidate_c_e_list = search_with_threshold(row["title_SS_embedding"], c_entity_title_vs, k=50)
            if len(candidate_c_e_list) >= 50: # we don't expect to hit this
                # essentially means we need to compact OR increase K OR rank better with summary etc.
                logger.error(f"50! candidate canonical entities found for raw entity {row['id']}. There are probably more that are getting missed.")
            # TODO SUBU Expand candidate selection with representative raw-entity embeddings.
            # - TODO SUBU identity search in known identities.
            # - vector search on known identities.
            # - add candidates from other text unit chunks for example.

            # FIND candidates
            # canonical OR TO-BE-CANONICAL nodes. Can contain duplicates.
            chosen_canonical_entity_ids: dict[Tuple[str, bool], CE_EntityHolder] = {} # map of (id, is_raw) -> CE_EntityHolder
            for ce in candidate_c_e_list:
                c_entity = c_graph_view.nodes[ce.document.id]
                # canonical nodes have relationships that have only summaries.
                # we want to ensure we catch all possible variants to ensure the match is good. So we figure the source raw entities and their relationships.
                relationships = []
                for raw_source in c_entity["raw_entity_ids"]:
                    relationships.extend([edge[2]["text_description"] for edge in r_graph.edges(raw_source, data=True)])
                
                chosen_canonical_entity_ids[(ce.document.id, False)] = CE_EntityHolder(
                    id=ce.document.id,
                    is_raw=False,
                    attributes=c_entity["attributes"],
                    title=c_entity["title"],
                    relationships=relationships,
                )
                # TODO SUBU - add the related entities as well
 
            # Search the new canonical entity prospects as well.
            if new_c_entity_prospects: # Chroma query filter searches everything for empty filter.
                r_entity_title_vs.filter_by_id(list(new_c_entity_prospects.keys()))
                candidate_r_e_list = search_with_threshold(row["title_SS_embedding"], r_entity_title_vs, k=50)
                if len(candidate_r_e_list) >= 50: # we don't expect to hit this in a single text unit
                    raise ValueError(f"50! candidate raw entities found for raw entity {row['id']}. There are probably more that are getting missed.")
                for re in candidate_r_e_list:
                    chosen_canonical_entity_ids[(str(re.document.id), True)] = new_c_entity_prospects[str(re.document.id)]

            if not chosen_canonical_entity_ids:
                # upgrade row to canonical entity
                # as time goes on, we expect this path to be rarer: We should know of all the canonical entities by now. TODO ADD METRIC
                chosen_canonical_entity_ids[(row["id"], True)] = raw_entity_holder

            # REFINE candidates
            canonicalization_result = None
            if len(chosen_canonical_entity_ids) == 1 and next(iter(chosen_canonical_entity_ids))[0] == row["id"] and next(iter(chosen_canonical_entity_ids))[1] == True:
                # we have only one raw candidate: self. we just upgrade it.
                canonicalization_result = CanonicalizationResult(
                    id=row["id"],
                    canonical_entities={(row["id"], True): {
                        "entity": raw_entity_holder,
                        "match_type": MatchType.EXACT,
                        "confidence": 1.0,
                        "reasoning": "Self-canonicalization",
                    }},
                )
            else:
                # handle canonical candidates, both new and existing: use context + LLM to refine to a smaller list OR even a new one.
                #
                # Right now this looks expensive: An LLM call per entity resolution. We will have to look at having smaller models and 
                # node similarity algos for this.
                # TODO SUBU NODE Similarlity is complicated because you want to map RAW->RAW connections on top of canonical->canonical connections and see if they are similar...
                # - So it is not a simple graph similarity algo - it is more vector similarity of edge descriptions (which usually include both sides of the edge).
                # - some are weighted higher (if there is a raw edge to a system entity - say the text mentions a profile url: This is huge compared to inferred edges)

                # TODO SUBU - we are doing a simplistic comparison based on text description of links, so we stop at depth 1 DFS. In an ideal world, 
                #  we will allow for target node details / summaries to be exactly sure. We will need enough links to 'WELL KNOWN' canonical entities to
                #  be able to reason about this.
                canonicalization_result = await canonicalize_entity(
                    r_entity=raw_entity_holder,
                    candidates=chosen_canonical_entity_ids,
                    cache=cache,
                    strategy=canonicalization_strategy,
                )
            found_exact_match = False
            # we get either
            # - exact match to an existing canonical entity(or prospect)
            # - partial match to an existing canonical entity(or prospect)
            # - no match to any canonical entity(or prospect)
            for k, v in canonicalization_result.canonical_entities.items():
                if k[1] == True: # any raw entity candidate is a prospect
                    new_c_entity_prospects[k[0]] = v["entity"]
                if v["match_type"] == MatchType.EXACT:
                    found_exact_match = True
            if not found_exact_match: # only partial or none. Upgrade current entity.
                new_c_entity_prospects[row["id"]] = raw_entity_holder

            raw_entity_to_canonical_entity_map[row["id"]] = canonicalization_result.canonical_entities
            # -- end thread


    # UPDATE system - create canonical entities and relationships (AFTER processing: outside the thread)
    def get_canonical_entity_row_for_raw_entity(raw_id: str) -> dict[str, Any]:
        raw_entity_node = r_graph.nodes[raw_id]
        return {
            "id": uuid7().hex,
            "title": raw_entity_node["title"],
            "title_SS_embedding": raw_entity_node["title_SS_embedding"],
            "metadata": {
                "attributes": raw_entity_node["attributes"],
                "node_type": SystemAttributes.CANONICAL,
            },
            "summary": "", # summary is generated later.
            "summary_RD_embedding": [],
            "raw_entity_ids": [raw_id],
        }
    r_to_c_entity_map = {} # map of raw_id -> canonical_entity_id
    # SINGLE THREADED NOW.
    # first pass to create canonical entities
    for raw_id, canonical_entity_map in raw_entity_to_canonical_entity_map.items():
        if len(canonical_entity_map) == 1 and next(iter(canonical_entity_map.items()))[1] == False:
            #single canonical candidate found
            ce_id = next(iter(canonical_entity_map.keys()))[0]
            # new HIERARCHY RELATIONSHIP from the raw entity
            ce_node = c_graph_view.nodes[ce_id]
            ce_node["raw_entity_ids"] = list(set(ce_node["raw_entity_ids"] + [raw_id]))
            # TODO SUBU improve this: For now we simplistically merge attributes. There can be unnecessary stuff like former/latter in there.
            ce_node["attributes"] = list(set(ce_node["attributes"] + raw_entity_map[raw_id].metadata["attributes"]))
        else:
            # we either have one raw entity candidate OR multiple ambiguous candidates.
            # Options (CHOSEN 2 for now):
            # 1. add ambiguous relationship from the raw entity to the canonical entities + confidence (This means the canonical relationships will need 
            # to be replicated to all of those ... and cleaned up as we disambiguate.)
            # 2. add a new canonical entity and add MAYBE_SAME_AS relationships to the chosen ones. This means we proliferate # of nodes 
            # -- more relationships than nodes. Easier to find clumps of ambiguous nodes.
            # -- can leave the node management to be the same process that merges duplicate canonical entities from parallel runs.
            # -- BUT confuses that process aswell: Parallel run dupes HAVE to be resolved (and probably can be because we haven't canonicalized across
            # the runs yet). We probably have no new information about this entity to help ... 
            # -- FWIW parallel runs are today based on vector k-means clustering. This means nodes from same document can be spread out.
            # --- BUT we are ok with this because new info could be doc comments, threads in the same channel or by same person / team etc ... it will get
            # disambiguated over time (or no one will care). Better to have a single disambiguation process than multiple semi-related ones.
            ce_row = get_canonical_entity_row_for_raw_entity(raw_id)
            ce_id = ce_row["id"]
            c_graph.add_node(
                ce_id,
                title=ce_row["title"],
                title_SS_embedding=ce_row["title_SS_embedding"],
                metadata=ce_row["metadata"],
                summary=ce_row["summary"],
                summary_RD_embedding=ce_row["summary_RD_embedding"],
                raw_entity_ids=ce_row["raw_entity_ids"],
            )
            # write to the vector store
            title_embedding = ce_row["title_SS_embedding"]
            if type(title_embedding) is np.ndarray:
                title_embedding = title_embedding.tolist()
            c_entity_title_vs.load_documents([VectorStoreDocument(
                id=ce_row["id"],
                text=ce_row["title"],
                vector=title_embedding,
                attributes={"title": ce_row["title"]}, # For some reason chroma is forcing this ... TODO SUBU FIX THIS
            )])

        r_to_c_entity_map[raw_id] = ce_id

    # second pass to canonicalize relationships from raw - now we have CE setup for all RE.
    # TODO SUBU Think about normalization(building known edge types) vs vector similarity matching (the current plan).
    for raw_id, ce_id in r_to_c_entity_map.items():
        for edge in r_graph.edges(raw_id, data=True):
            source_ce = r_to_c_entity_map[edge[0]]
            target_ce = r_to_c_entity_map[edge[1]]
            if c_graph.has_edge(source_ce, target_ce):
                # NOTE: canonical descriptions get summarized periodically. For now we just put a marker
                c_graph.edges[source_ce, target_ce]["metadata"]["canonical_summary_pending"] = True 
                c_graph.edges[source_ce, target_ce]["weight"] += edge[2]["weight"]
            else:
                c_graph.add_edge(
                    source_ce,
                    target_ce,
                    weight=edge[2]["weight"],
                    metadata={
                        "edge_type": SystemAttributes.CANONICAL,
                        "relationship_type": RelationshipType.RELATES_TO,
                        "canonical_summary_pending": True,
                        "last_summary_timestamp": 0,
                    },
                    canonical_summary="",
                    raw_descriptions=[],
                    canonical_summary_RD_embedding=[], # needs to be filled in later in generate embeddings step.
                )
    # save partial match info
    for raw_id, canonical_entity_map in raw_entity_to_canonical_entity_map.items():
        ce_id = r_to_c_entity_map[raw_id]
        for (c_r_id, is_raw), llm_result in canonical_entity_map.items():
            if is_raw:
                c_ce_id = r_to_c_entity_map[c_r_id]
            else:
                c_ce_id = c_r_id
            if llm_result["match_type"] != MatchType.PARTIAL:
                c_graph.add_edge(
                    ce_id,
                    c_ce_id,
                    weight=1.0,
                    metadata={
                        "edge_type": SystemAttributes.CANONICAL,
                        "relationship_type": RelationshipType.PARTIAL_MATCH,
                        "confidence": llm_result["confidence"],
                        "reasoning": llm_result["reasoning"],
                        "canonical_summary_pending": True,
                        "last_summary_timestamp": 0,
                    },
                )
            elif llm_result["match_type"] == MatchType.NONE:
                c_graph.add_edge(
                    ce_id,
                    c_ce_id,
                    weight=1.0,
                    metadata={
                        "edge_type": SystemAttributes.CANONICAL,
                        "relationship_type": RelationshipType.NO_MATCH,
                        "reasoning": llm_result["reasoning"],
                    },
                )

    canonical_entities = pd.DataFrame([
        ({"id": item[0], **(item[1] or {})})
        for item in c_graph.nodes(data=True)
        if item is not None
    ])
    canonical_relationships = pd.DataFrame(nx.to_pandas_edgelist(c_graph, edge_key="key"))

    return canonical_entities, canonical_relationships






