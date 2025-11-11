# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
<Goal>
Given a text document of type {document_type}, identify all nouns and concepts as entities. Capture their name, attributes and their types from the text. Also extract relationships among the identified entities.
</Goal>
<Steps>
<Step id="1">
Identify all nouns and concepts entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized. Resolve references within the text to the right entity. Don't create entities that cannot be identified by name.
- entity_id: create an unique id out of the name if there are duplicate entries. Else use entity_name itself
- entity_attributes: List of the entity's attributes that specifically identifies this instance. Do not include information that can be modeled as a relationship with another entity. Leave empty if nothing is provided.
- entity_type: Pick an appropriate entity type based on the entity's name and attributes. If unsure, use "ENTITY".
</Step>
<Step id="2">
From the entities identified in step 1, identify all pairs of (source entity, target entity) that are *clearly related* to each other. The relationships are bidirectional.
For each pair of related entities, extract the following information:
- source_entity_id: id of the source entity, as identified in step 1
- target_entity_id: id of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other based on the text. Do not include information not present in the text.
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
</Step>
</Step>
<Step id="3">

Return output as a well-formed JSON-formatted string with provided format. Don't use any unnecessary escape sequences. The output should be a single JSON object that can be parsed by json.loads.

</Step>
</Steps>

<Examples>
<Example>
<Input>
Document_type: NEWS
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
</Input>
<Output>
{
 "entities": [
  { 
  "entity_name": "Verdantis's Central Institution",
  "entity_id": "Verdantis's Central Institution",
   "entity_type": "ORGANIZATION",
   "entity_attributes": []
  },
  {
   "entity_name": "Monday",
"entity_id": "Monday",
   "entity_type": "DATE",
   "entity_attributes": []
  },
  {
   "entity_name": "Thursday",
"entity_id": "Thursday",
   "entity_type": "DATE",
   "entity_attributes": []
  },
  {
   "entity_name": "Latest policy decision",
"entity_id": "Latest policy decision",
   "entity_type": "CONCEPT",
   "entity_attributes": []
  },
  {
   "entity_name": "1:30 p.m. PDT",
"entity_id": "1:30 p.m. PDT",
   "entity_type": "TIME",
   "entity_attributes": []
  },
  {
   "entity_name": "Press conference",
"entity_id": "Press conference",
   "entity_type": "EVENT",
   "entity_attributes": []
  },
  {
   "entity_name": "Martin Smith",
"entity_id": "Martin Smith",
   "entity_type": "PERSON",
   "entity_attributes": ["Chair"]
  },
  {
   "entity_name": "Verdantis's Central Institution Chair Martin Smith",
"entity_id": "Verdantis's Central Institution Chair Martin Smith",
   "entity_type": "PERSON",
   "entity_attributes": []
  },
  {
   "entity_name": "Investors",
"entity_id": "Investors",
   "entity_type": "PERSON_GROUP",
   "entity_attributes": []
  },
  {
   "entity_name": "Market Strategy Committee",
"entity_id": "Market Strategy Committee",
   "entity_type": "ORGANIZATION",
   "entity_attributes": []
  },
  {
   "entity_name": "Benchmark interest rate",
"entity_id": "Benchmark interest rate",
   "entity_type": "FINANCIAL_INSTRUMENT",
   "entity_attributes": []
  },
  {
   "entity_name": "3.5%-3.75%",
"entity_id": "3.5%-3.75%",
   "entity_type": "PERCENTAGE",
   "entity_attributes": []
  }
 ],
 "relationships": [
  {
   "source_entity_id": "Verdantis's Central Institution",
"target_entity_id": "Monday",
   "relationship_description": "The Verdantis's Central Institution is scheduled to meet on Monday.",
   "relationship_strength": 0.8
  },
  {
   "source_entity_id": "Verdantis's Central Institution",
   "target_entity_id": "Thursday",
   "relationship_description": "The Verdantis's Central Institution is scheduled to meet on Thursday.",
   "relationship_strength": 0.8
  },
  {
   "source_entity_id": "Verdantis's Central Institution",
   "target_entity_id": "Latest policy decision",
   "relationship_description": "The Verdantis's Central Institution is planning to release its latest policy decision.",
   "relationship_strength": 0.7
  },
  {
   "source_entity_id": "Latest policy decision",
   "target_entity_id": "Thursday",
   "relationship_description": "The latest policy decision is to be released on Thursday.",
   "relationship_strength": 0.8
  },
  {
   "source_entity_id": "Latest policy decision",
   "target_entity_id": "1:30 p.m. PDT",
   "relationship_description": "The latest policy decision is to be released at 1:30 p.m. PDT.",
   "relationship_strength": 0.8
  },
  {
   "source_entity_id": "Verdantis's Central Institution",
   "target_entity_id": "Press conference",
   "relationship_description": "The release of the policy decision is followed by a press conference hosted by The Verdantis's Central Institution.",
   "relationship_strength": 0.7
  },
  {
   "source_entity_id": "Press conference",
   "target_entity_id": "Verdantis's Central Institution Chair Martin Smith",
   "relationship_description": "Verdantis's Central Institution Chair Martin Smith will take questions at the press conference.",
   "relationship_strength": 0.9
  },
  {
   "source_entity_id": "Martin Smith",
   "target_entity_id": "Verdantis's Central Institution",
   "relationship_description": "Martin Smith is the Chair of the Verdantis's Central Institution.",
   "relationship_strength": 0.9
  },
  {
   "source_entity_id": "Investors",
   "target_entity_id": "Market Strategy Committee",
   "relationship_description": "Investors expect the Market Strategy Committee to take action.",
   "relationship_strength": 0.6
  },
  {
   "source_entity_id": "Market Strategy Committee",
   "target_entity_id": "Benchmark interest rate",
   "relationship_description": "The Market Strategy Committee is expected to hold its benchmark interest rate steady.",
   "relationship_strength": 0.8
  },
  {
   "source_entity_id": "Benchmark interest rate",
   "target_entity_id": "3.5%-3.75%",
   "relationship_description": "The benchmark interest rate is expected to be in the range of 3.5%-3.75%.",
   "relationship_strength": 0.9
  }
 ]
}
</Output>
</Example>
<Example>
<Input>
Document_type: NEWS
Text:
Rama and Rama are two friends with the same name. The former married Sita and latter lived with Lash.
</Input>
<Output>
{
  "entities": [
    {
      "entity_name": "Rama",
      "entity_id": "Rama 1",
      "entity_type": "PERSON",
      "entity_attributes": [
        "former"
      ]
    },
    {
      "entity_name": "Rama",
      "entity_id": "Rama 2",
      "entity_type": "PERSON",
      "entity_attributes": [
        "latter"
      ]
    },
    {
      "entity_name": "Two friends",
      "entity_id": "Two friends",
      "entity_type": "PERSON_GROUP",
      "entity_attributes": [
        "with the same name"
      ]
    },
    {
      "entity_name": "Sita",
      "entity_id": "Sita",
      "entity_type": "PERSON",
      "entity_attributes": []
    },
    {
      "entity_name": "Lash",
      "entity_id": "Lash",
      "entity_type": "PERSON",
      "entity_attributes": []
    }
  ],
  "relationships": [
    {
      "source_entity_id": "Rama 1",
      "target_entity_id": "Two friends",
      "relationship_description": "Rama 1 is one of the two friends.",
      "relationship_strength": 0.9
    },
    {
      "source_entity_id": "Rama 2",
      "target_entity_id": "Two friends",
      "relationship_description": "Rama 2 is one of the two friends.",
      "relationship_strength": 0.9
    },
    {
      "source_entity_id": "Rama 1",
      "target_entity_id": "Sita",
      "relationship_description": "Rama 1 married Sita.",
      "relationship_strength": 0.9
    },
    {
      "source_entity_id": "Rama 2",
      "target_entity_id": "Lash",
      "relationship_description": "Rama 2 lived with Lash.",
      "relationship_strength": 0.8
    }
  ]
}
</Output>
</Example>
</Examples>

Document_type: {document_type}
Text: {input_text}
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
