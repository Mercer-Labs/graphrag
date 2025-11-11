from pydantic import BaseModel, Field


class RawEntityModel(BaseModel):
    """A model for an entity."""

    entity_name: str = Field(description="The name of the entity.")
    entity_type: str = Field(description="The type of the entity.")
    entity_attributes: list[str] = Field(description="The attributes of the entity.")


class RawRelationshipModel(BaseModel):
    """A model for a relationship."""

    source_entity: str = Field(description="The name of the source entity.")
    target_entity: str = Field(description="The name of the target entity.")
    relationship_description: str = Field(description="The description of the relationship.")
    relationship_strength: float = Field(description="The strength of the relationship.")

class ExtractGraphResponse(BaseModel):
    """A model for the expected LLM response shape."""

    entities: list[RawEntityModel] = Field(description="A list of entities identified.")
    relationships: list[RawRelationshipModel] = Field(description="A list of relationships identified.")


