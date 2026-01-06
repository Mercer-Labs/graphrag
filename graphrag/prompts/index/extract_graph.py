# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
<Goal>
Given a text document of type {document_type}, identify all nouns and concepts as entities. Capture their name, attributes and their types from the text. Also extract relationships among the identified entities.
</Goal>
<Rules>
- Don't decode known types of entities like Phone numbers,URLs, email addresses etc: Treat them as entities with corresponding types.
- Treat self references like me, my, we, us, etc. as entities with types like SELF_PERSON_REFERENCE, SELF_ORG_REFERENCE etc.
</Rules>
<Steps>
<Step id="1">
Identify all nouns and concepts entities. For each identified entity, extract the following information:
- name: Name of the entity, capitalized. Resolve references within the text to the right entity. Don't create entities that cannot be identified by name.
- id: create an unique id out of the name if there are duplicate entries. Else use name itself
- attributes: List of the entity's attributes that specifically identifies this instance. Do not include information that can be modeled as a relationship with another entity. Leave empty if nothing is provided.
- type: Pick an appropriate entity type based on the entity's name and attributes. If unsure, use "ENTITY".
- is_proper_noun: Is this entity referring to a specific person, place, or thing? If unsure, set to False.
</Step>
<Step id="2">
From the entities identified in step 1, identify all pairs of (source entity, target entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source: id of the source entity, as identified in step 1
- target: id of the target entity, as identified in step 1
- text_location: the start location index of the relationship in the provided text.
- description: explanation as to why you think the source entity and the target entity are related to each other based on the text. Do not include information not present in the text. Escape any usage of $ or # in the description. When referencing entities in the description, use $##1 and $##2 placeholders to reference the source and target entities respectively.
- strength: a numeric score indicating strength of the relationship between the source entity and target entity
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
Rama and Rama are two friends with the same name. The former married Sita and latter lived with Lash.
</Input>
<Output>
{{
 "entities": [
  {{
   "name": "Rama",
   "id": "Rama 1",
   "type": "PERSON",
   "attributes": [
    "former"
   ]
  }},
  {{
   "name": "Rama",
   "id": "Rama 2",
   "type": "PERSON",
   "attributes": [
    "latter"
   ]
  }},
  {{
   "name": "Two friends",
   "id": "Two friends",
   "type": "PERSON_GROUP",
   "attributes": [
    "with the same name"
   ]
  }},
  {{
   "name": "Sita",
   "id": "Sita",
   "type": "PERSON",
   "attributes": []
  }},
  {{
   "name": "Lash",
   "id": "Lash",
   "type": "PERSON",
   "attributes": []
  }}
 ],
 "relationships": [
  {{
   "source": "Rama 1",
   "target": "Two friends",
   "description": "$##1 is one of the $##2.",
   "text_location": 1,
   "strength": 0.9
  }},
  {{
   "source": "Rama 2",
   "target": "Two friends",
   "description": "$##1 is one of the $##2.",
   "text_location": 1,
   "strength": 0.9
  }},
  {{
   "source": "Rama 1",
   "target": "Sita",
   "description": "$##1 married $##2.",
   "text_location": 51,
   "strength": 0.9
  }},
  {{
   "source": "Rama 2",
   "target": "Lash",
   "description": "$##1 lived with $##2.",
   "text_location": 51,
   "strength": 0.8
  }}
 ]
}}
</Output>
</Example>
<Example>
<Input>
Document_type: NEWS
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT. It’s Chair Martin Smith will take questions after.
</Input>
<Output>
{{
  "entities": [
    {{
      "name": "Verdantis's Central Institution",
      "id": "Verdantis's Central Institution",
      "type": "ORGANIZATION",
      "attributes": []
    }},
    {{
      "name": "Monday",
      "id": "Monday",
      "type": "DATE",
      "attributes": []
    }},
    {{
      "name": "Thursday",
      "id": "Thursday",
      "type": "DATE",
      "attributes": []
    }},
    {{
      "name": "Institution",
      "id": "Institution",
      "type": "ORGANIZATION",
      "attributes": [],
      "resolved_reference": "Verdantis's Central Institution"
    }},
    {{
      "name": "Policy decision",
      "id": "Policy decision",
      "type": "CONCEPT",
      "attributes": [
        "latest"
      ]
    }},
    {{
      "name": "1:30 p.m. PDT",
      "id": "1:30 p.m. PDT",
      "type": "TIME",
      "attributes": []
    }},
    {{
      "name": "Chair Martin Smith",
      "id": "Chair Martin Smith",
      "type": "PERSON",
      "attributes": [
        "Chair"
      ]
    }},
    {{
      "name": "Questions",
      "id": "Questions",
      "type": "CONCEPT",
      "attributes": []
    }}
  ],
  "relationships": [
    {{
      "source": "Verdantis's Central Institution",
      "target": "Monday",
      "description": "$##1 is scheduled to meet on $##2.",
      "text_location": 36,
      "strength": 0.9
    }},
    {{
      "source": "Verdantis's Central Institution",
      "target": "Thursday",
      "description": "$##1 is scheduled to meet on $##2.",
      "text_location": 47,
      "strength": 0.9
    }},
    {{
      "source": "Institution",
      "target": "Policy decision",
      "description": "$##1 is planning to release its $##2.",
      "text_location": 73,
      "strength": 0.9
    }},
    {{
      "source": "Institution",
      "target": "Thursday",
      "description": "$##1 is planning to release its latest policy decision on $##2.",
      "text_location": 95,
      "strength": 0.9
    }},
    {{
      "source": "Institution",
      "target": "1:30 p.m. PDT",
      "description": "$##1 is planning to release its latest policy decision at $##2.",
      "text_location": 107,
      "strength": 0.9
    }},
    {{
      "source": "Chair Martin Smith",
      "target": "Institution",
      "description": "$##1 is the Chair of the $##2.",
      "text_location": 122,
      "strength": 0.8
    }},
    {{
      "source": "Chair Martin Smith",
      "target": "Questions",
      "description": "$##1 will take $##2 after the policy decision release.",
      "text_location": 138,
      "strength": 0.8
    }}
  ]
}}
</Output>
</Example>
</Examples>

Document_type: {document_type}
Text: {input_text}
Output:"""

CONTINUE_PROMPT = "Some entities and relationships were missed in the last extraction. Add them below using the same format. Remember to ONLY emit missing entities and relationships\n"
LOOP_PROMPT = "Answer Y if there are more entities or relationships that need to be extracted beyond the maximum allowed, or N if there are none. Please answer with a single letter Y or N.\n"