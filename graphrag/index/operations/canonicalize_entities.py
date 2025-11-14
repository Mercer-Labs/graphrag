# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to canonicalize the entities. We want to come out of this step with fully identified canonical entities for a given entity."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans
from uuid_utils import uuid7

from graphrag.config.embeddings import (
    canonical_entity_description_embedding,
    canonical_entity_title_embedding,
    raw_entity_title_embedding,
)
from graphrag.data_model.schemas import SystemAttributes
from graphrag.index.operations.embed_text.embed_text import get_vector_store_for_write
from graphrag.utils.api import get_vector_store_for_query
from graphrag.vector_stores.base import BaseVectorStore

logger = logging.getLogger(__name__)

# slightly generous: around .15 seems to include plurals etc. But leaving it here for better consolidation with more context.
VECTOR_SEARCH_DISTANCE_THRESHOLD = 0.2


def search_with_threshold(embedding, vector_store: BaseVectorStore, k: int = 5, threshold: float = VECTOR_SEARCH_DISTANCE_THRESHOLD):
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
def canonicalize_entities(
    raw_entities: pd.DataFrame,
    raw_relationships: pd.DataFrame,
    known_identities: pd.DataFrame,
    known_relationships: pd.DataFrame,
    text_embed_config_strategy: dict,
    num_threads: int = 4,
    canonical_entities: pd.DataFrame | None = None,
    canonical_relationships: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
    # TODO SUBU we assume there are num_threads clusters. In reality there may not be: Look at silhouette score/analysis to pick this number better.
    kmeans = KMeans(n_clusters=num_threads, random_state=42)
    k_clusters = kmeans.fit(raw_entities["embedding"].tolist()) # Probably a better way to do this...
    raw_entities["cluster"] = k_clusters.labels_
    raw_entity_cluster_dfs = {cluster: group for cluster, group in raw_entities.groupby("cluster")}


    if canonical_entities is None:
        canonical_entities = pd.DataFrame(columns=["id", "title", "title_embedding", "metadata", "summary", "summary_embedding", "representative_raw_entity_ids"])
    if canonical_relationships is None:
        canonical_relationships = pd.DataFrame(columns=["id", "source", "target", "metadata", "summary", "summary_embedding"])

    logger.debug(f"Vector Store Args: {text_embed_config_strategy['vector_store']}")
    c_entity_title_vs = get_vector_store_for_write(
        vector_store_config=text_embed_config_strategy['vector_store'],
        llm_config=text_embed_config_strategy['llm'],
        embedding_name=canonical_entity_title_embedding,
    )

    # NOTE Because of the parallelization, we cannot avoid duplicate canonical entities getting created. So we need a compaction round to consolidate 
    # them. The compaction needs to be done with context and probably graph shape analysis AND offline. Once this is cross machine, the simplistic finalization
    # here won't be enough. To reduce the duplicates as much as possible, we do serialize within each thread here.
    # 
    # NOTE: This merging is what makes the representative raw-entity embeddings very useful for future canonicalization. It tries to bring the LLM intel to 
    # embedding vectors for name searches.
    # TODO SUBU MOVE THIS WITHIN THE THREAD
    raw_entity_to_canonical_entity_map = {} # map of raw_id -> set(canonical_entity_ids)
    new_canonical_prospects = {} # map of raw_id -> title 
    #(TODO SUBU FOR DEBUGGING)
    raw_e_map = {row.id: row.title for row in raw_entities.itertuples(index=False)}

    for cluster, cluster_df in raw_entity_cluster_dfs.items():
        logger.debug(f"Processing cluster {cluster} with {len(cluster_df)} raw entities.")
        # TODO SUBU PARALLELIZE the below

        # init vectore store per thread: TO allow for query filters
        # since we are processing raw entities from THIS run / this vector store, we will use embed_text_vector_store_id to identify the store
        r_entity_title_vs = get_vector_store_for_query(
            vector_store_args={
                text_embed_config_strategy['embed_text_vector_store_id']: text_embed_config_strategy['vector_store'],
            },
            llm_config=text_embed_config_strategy['llm'],
            embedding_name=raw_entity_title_embedding,
        )

        #  we maintain a list of the prospects instead of directly creating canonical entities here. Because without canonical edges, these are
        #  not useful for context checks - will only confuse the LLM.
        for row in cluster_df.itertuples(index=False):
            logger.debug(f"Processing raw entity {row.id} in cluster {cluster}.")
            # find candidate canonical entities by vector search on canonical entities.
            candidate_c_e_list = search_with_threshold(row.embedding, c_entity_title_vs)
            # TODO SUBU Expand candidate selection with representative raw-entity embeddings.
            # - TODO SUBU identity search in known identities.
            # - vector search on known identities.

            # FIND candidates
            # note - this can contain duplicates.
            chosen_canonical_entity_ids = set() #  (id, is_raw)
            if candidate_c_e_list:
                chosen_canonical_entity_ids.update([(ce.document.id, False) for ce in candidate_c_e_list])
            else:
                # as time goes on, we expect this path to be rarer: We should know of all the canonical entities by now. TODO ADD METRIC
                pass
            # search in the new prospects as well to converge.
            if new_canonical_prospects:
                r_entity_title_vs.filter_by_id(list(new_canonical_prospects.keys()))
                candidate_r_e_list = search_with_threshold(row.embedding, r_entity_title_vs)
                if candidate_r_e_list:
                    chosen_canonical_entity_ids.update([(re.document.id, True) for re in candidate_r_e_list])

            if not chosen_canonical_entity_ids:
                chosen_canonical_entity_ids.add((row.id, True))

            # REFINE candidates
            # NOTE The fancy any is just to access the first element of the set.
            if len(chosen_canonical_entity_ids) == 1 and any(is_raw for _, is_raw in chosen_canonical_entity_ids):
                # we have only one raw candidate: self. we just upgrade it.
                ((e_id, is_raw)) = chosen_canonical_entity_ids
                assert is_raw and e_id == row.id, "Raw entity should be the only candidate."
            else:
                # handle canonical candidates, both new and existing: use context + LLM to refine to a smaller list OR even a new one.
                #
                # Right now this looks expensive: An LLM call per entity resolution. We will have to look at having smaller models and 
                # node similarity algos for this.
                # TODO SUBU NODE Similarlity is complicated because you want to map RAW->RAW connections on top of canonical->canonical connections and see if they are similar...
                # - So it is not a simple graph similarity algo - it is more vector similarity of edge descriptions (which usually include both sides of the edge).
                # - some are weighted higher (if there is a raw edge to a system entity - say the text mentions a profile url: This is huge compared to inferred edges)
                pass

            # UPDATE system
            if len(chosen_canonical_entity_ids) == 1:
                # add stable hierarchy relationships from the raw entity to the canonical entity.
                pass
            else:
                # add ambiguous relationship from the raw entity to the canonical entities + confidence
                pass
            raw_entity_to_canonical_entity_map[row.id] = chosen_canonical_entity_ids
            for e_id, is_raw in chosen_canonical_entity_ids:
                if is_raw:
                    new_canonical_prospects[e_id] = raw_e_map[e_id]

    # create the canonical entities and relationships.

    canonical_entities = canonical_entities.reset_index(drop=True)
    return canonical_entities





                # new_canonical_entity = [
                #     uuid7().hex, # use this for time-sortability
                #     row.title, # TODO SUBU create a canonical title from the raw entity title.
                #     row.embedding, 
                #     {
                #         "attributes": row.attributes, # TODO SUBU - filter this to keep only identifying attributes.
                #         "entity_type": row.llm_type,
                #         "is_proper_noun": row.is_proper_noun,
                #         "node_type": SystemAttributes.CANONICAL,
                #     },
                #     "", # summary is generated later.
                #     [],
                # ]
                # canonical_entities.loc[len(canonical_entities)] = new_canonical_entity
                #
                # add a relationship from the raw entity to the canonical entity + similarity score


    # Canonical entities can have multiple representative raw-entity embeddings.
    # Today we simplistically search the canonical db TODO AND raw db (with metadata check for CANONICAL_REPRESENTATIVE flag)

    # we maintain max of 5 representative raw-entity embeddings per canonical entity. HERE we do the flag maintenance.
    # we need to cap this number otherwise the raw search will result in too many results that resolve to same canonical entity.
    # So we want to keep the flag on the 'most different' raw-entity embeddings.
    # TODO LATER this probably can be done better - but we will move to db anyway
    # repr_embeddings = canonical_entities[["id", "repr_raw_entity_ids"]].explode(
    #     "repr_raw_entity_ids"
    # ).merge(
    #     raw_entities[["id", "embedding"]],
    #     left_on="repr_raw_entity_ids",
    #     right_on="id",
    #     how="inner",
    #     suffixes=("_canonical", "_raw"),
    # ).groupby("id_canonical").agg(repr_embeddings=("embedding", list)).reset_index()
    
    # canonical_entities = canonical_entities.merge(
    #     repr_embeddings,
    #     left_on="id",
    #     right_on="id_canonical",
    #     how="left",
    # )
    # IF there are < 5, just add. Else do a dispersion check and keep the most dispersed.

