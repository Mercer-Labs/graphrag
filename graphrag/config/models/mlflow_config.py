# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import mlflow_defaults


class MLFlowConfig(BaseModel):
    """Configuration section for MLflow."""

    enabled: bool = Field(
        description="A flag indicating whether to enable MLflow tracking.",
        default=mlflow_defaults.enabled,
    )

    tracking_uri: str = Field(
        description="The tracking URI to use for MLflow.",
        default=mlflow_defaults.tracking_uri,
    )

    experiment_name: str = Field(
        description="The experiment name to use for MLflow.",
        default=mlflow_defaults.experiment_name,
    )
