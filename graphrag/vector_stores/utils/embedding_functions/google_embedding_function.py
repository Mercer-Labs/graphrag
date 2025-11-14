import os
from typing import Any, Dict, List, Optional, cast

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space
from chromadb.utils.embedding_functions.schemas import validate_config_schema


# NOTE custom copy from chroma-core because 
# - support output dimensionality config
# - support direct api_key
class GoogleGenaiEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        task_type: str,
        vertexai: Optional[bool] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize the GoogleGenaiEmbeddingFunction.

        Args:
            model_name (str): The name of the model to use for text embeddings.
            api_key_env_var (str, optional): Environment variable name that contains your API key for the Google GenAI API.
                Defaults to "GOOGLE_API_KEY".
        """
        try:
            import google.genai as genai
            import google.genai.types as types
        except ImportError:
            raise ValueError(
                "The google-genai python package is not installed. Please install it with `pip install google-genai`"
            )

        self.model_name = model_name
        self.vertexai = vertexai
        self.project = project
        self.location = location
        self.api_key = api_key
        self.task_type = task_type
        if not self.api_key:
            raise ValueError("null api_key provided")
        self.dimensions = dimensions
        if self.dimensions is not None:
            self.embed_config = types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
                task_type=self.task_type
                )
        else:
            self.embed_config = types.EmbedContentConfig()

        self.client = genai.Client(
            api_key=self.api_key, vertexai=vertexai, project=project, location=location
        )

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.

        Args:
            input: Documents or images to generate embeddings for.

        Returns:
            Embeddings for the documents.
        """
        if not input:
            raise ValueError("Input documents cannot be empty")
        if not isinstance(input, (list, tuple)):
            raise ValueError("Input must be a list or tuple of documents")
        if not all(isinstance(doc, str) for doc in input):
            raise ValueError("All input documents must be strings")

        try:
            response = self.client.models.embed_content(
                model=self.model_name, contents=input, config=self.embed_config
            )
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {str(e)}") from e

        # Validate response structure
        if not hasattr(response, "embeddings") or not response.embeddings:
            raise ValueError("No embeddings returned from the API")

        embeddings_list = []
        for ce in response.embeddings:
            if not hasattr(ce, "values"):
                raise ValueError("Malformed embedding response: missing 'values'")
            embeddings_list.append(np.array(ce.values, dtype=np.float32))

        return cast(Embeddings, embeddings_list)

    @staticmethod
    def name() -> str:
        return "google_genai"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        model_name = config.get("model_name")
        api_key = config.get("api_key")
        vertexai = config.get("vertexai")
        project = config.get("project")
        location = config.get("location")
        task_type = config.get("task_type")
        dimensions = config.get("dimensions")

        if model_name is None:
            raise ValueError("The model name is required.")

        return GoogleGenaiEmbeddingFunction(
            model_name=model_name,
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            task_type=task_type,
            dimensions=dimensions,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "vertexai": self.vertexai,
            "project": self.project,
            "location": self.location,
            "task_type": self.task_type,
            "dimensions": self.dimensions,
        }

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        if "model_name" in new_config:
            raise ValueError(
                "The model name cannot be changed after the embedding function has been initialized."
            )
        if "vertexai" in new_config:
            raise ValueError(
                "The vertexai cannot be changed after the embedding function has been initialized."
            )
        if "project" in new_config:
            raise ValueError(
                "The project cannot be changed after the embedding function has been initialized."
            )
        if "location" in new_config:
            raise ValueError(
                "The location cannot be changed after the embedding function has been initialized."
            )

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate the configuration using the JSON schema.

        Args:
            config: Configuration to validate

        Raises:
            ValidationError: If the configuration does not match the schema
        """
        validate_config_schema(config, "google_genai")

