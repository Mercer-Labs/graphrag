# GraphRAG Setup
from typing import Any

import chromadb
import numpy as np
from chromadb.utils.batch_utils import create_batches

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.data_model.types import TextEmbedder

# GraphRAG vector store components
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)
from graphrag.vector_stores.utils.embedding_functions.google_embedding_function import (
    GoogleGenaiEmbeddingFunction,
)


# NOTE this is just to be sure - we don't generate embeddings from GraphRAG right now.
# TODO SUBU Lot of type warnings for some reason - fix later.
class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation
    """
    persistent_client: str

    def __init__(self, vector_store_schema_config: VectorStoreSchemaConfig, llm_config: dict, **kwargs: Any):
        """Initialize the in-memory vector store."""
        super().__init__(vector_store_schema_config=vector_store_schema_config, **kwargs)
        if not self.index_name:
            raise ValueError("index_name is required in vector_store_schema_config")

        if "custom_parameters" not in kwargs or not kwargs["custom_parameters"]:
            raise ValueError("custom_parameters is required")
        custom_parameters = kwargs["custom_parameters"]

        if "persistent_client" not in custom_parameters or not custom_parameters["persistent_client"]:
            raise ValueError("persistent_client is required")
        self.persistent_client = custom_parameters["persistent_client"]
        self.google_ef = GoogleGenaiEmbeddingFunction(
            api_key=llm_config["api_key"], 
            task_type=self.task_type,
            model_name=llm_config["model"],
            dimensions=self.vector_size
        )

    def connect(self, **kwargs: Any) -> None:
        """Connect to the vector storage (no-op for in-memory store)."""
        self.db_connection = chromadb.PersistentClient(self.persistent_client)
        self.document_collection = self.db_connection.get_or_create_collection(
            self.index_name,
            embedding_function=self.google_ef,
        )

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """
        Load documents into the vector store.
        NOTE: This expects the documents to have already been embedded. Else it will be ignored.
        """
        if not self.db_connection:
            msg = "Vector store not connected. Call connect() first."
            raise RuntimeError(msg)

        if overwrite and self.index_name in self.db_connection.list_collections():
            self.db_connection.delete_collection(self.index_name)
            self.document_collection = self.db_connection.create_collection(
                self.index_name,
                embedding_function=self.google_ef,
            )

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for document in documents:
            if document.vector is not None and len(document.vector) == self.vector_size:
                ids.append(document.id)
                texts.append(document.text)
                embeddings.append(np.array(document.vector, dtype=np.float32))
                metadatas.append(document.attributes)
        
        if len(ids) != len(documents):
            print(f"WARNING: Some documents were not embedded. Skipping them.")
        
        self._add_documents(ids, texts, embeddings, metadatas)

    def _add_documents(self, ids, documents, embeddings, metadatas):
        batches = create_batches(
            api=self.db_connection,ids=list(ids),
            documents=list(documents),
            embeddings=list(embeddings) if embeddings else None, 
            metadatas=list(metadatas))
        for batch in batches:
            self.document_collection.add(
                ids=batch[0],
                documents=batch[3],
                embeddings=batch[1],
                metadatas=batch[2])

    # Some helpers for other non GraphRAG vector operations
    def add_document(self, document: str, id: str, metadata: dict[str, Any]) -> None:
        """Add a document to the vector store.
        """
        if not self.db_connection:
            msg = "Vector store not connected. Call connect() first."
            raise RuntimeError(msg)
        
        self.document_collection.add(
            ids=[id],
            documents=[document],
            metadatas=[metadata]
        )
    
    # Some helpers for other non GraphRAG vector operations
    def add_documents(self, documents: list[str], ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Add a list of documents to the vector store."""
        if not self.db_connection:
            msg = "Vector store not connected. Call connect() first."
            raise RuntimeError(msg)
        
        self._add_documents(ids, documents, None, metadatas)
        
    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for the vector store."""
        return self.db_connection.get_max_batch_size()

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform similarity search using a query vector."""
        if not self.db_connection:
            msg = "Vector store not connected. Call connect() first."
            raise RuntimeError(msg)

        query_vec = np.array(query_embedding, dtype=np.float32)
        top_k = self.document_collection.query(
            query_embeddings=query_vec,
            n_results=k,
            include=["documents", "metadatas", "distances", "embeddings"],
            ids=self.query_filter
        )
        if top_k is None:
            return []

        if len(top_k["ids"]) != 1:
            raise ValueError(f"Expected 1 result, got {len(top_k['ids'])}")
        
        # Create search results
        results = []
        for i in range(len(top_k["ids"][0])):
            document = VectorStoreDocument(
                id=top_k["ids"][0][i],
                text=top_k["documents"][0][i],
                vector=top_k["embeddings"][0][i],
                attributes=top_k["metadatas"][0][i]
            )
            result = VectorStoreSearchResult(document=document, score=top_k["distances"][0][i])
            results.append(result)

        return results

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform similarity search using text (which gets embedded first)."""
        # Embed the text first
        query_embedding = text_embedder(text)

        # Use vector search with the embedding
        return self.similarity_search_by_vector(query_embedding, k, **kwargs)

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            if isinstance(include_ids[0], str):
                self.query_filter = include_ids
            else:
                self.query_filter = [str(id) for id in include_ids]
        return self.query_filter


    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for a document by id."""
        doc = self.document_collection.get(ids=[id], include=["documents", "metadatas", "embeddings"])
        if doc:
            if len(doc["ids"]) > 1:
                raise ValueError(f"Multiple documents found for id {id}")
            return VectorStoreDocument(
                id=doc["ids"][0],
                text=doc["documents"][0],
                vector=doc["embeddings"][0],
                attributes=doc["metadatas"][0],
            )
        return VectorStoreDocument(id=id, text=None, vector=None)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store (custom method)."""
        return {
            "index_name": self.index_name,
            "document_count": self.document_collection.count(),
            "vector_count": self.document_collection.count(),
            "connected": self.db_connection is not None,
            "vector_dimension": self.vector_size, # See comment above - this can be wrong.
        }

