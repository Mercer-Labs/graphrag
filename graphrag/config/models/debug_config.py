# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from dataclasses import asdict

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults, mlflow_defaults
from graphrag.config.models.mlflow_config import MLFlowConfig


class DebugConfig(BaseModel):
    """Configuration section for debug."""

    mlflow: MLFlowConfig = Field(
        description="The MLflow configuration.",
        default=MLFlowConfig(**asdict(mlflow_defaults)),
    )
