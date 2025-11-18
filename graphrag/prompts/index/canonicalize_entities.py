# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition for entity canonicalization."""

# TODO SUBU compact the json inputs. I wonder if LLMs will do better if it was in story form anyway
CANONICALIZE_ENTITY_PROMPT = """
You are a helpful assistant responsible for deciding whether a new entity is the same as any of the canonical entities in an entity relationship graph.

# GOAL
Find the best matching canonical entity for a given new entity. Provide the confidence score and reasoning. The goal is to find out if the current entity refers to the same real-world entity as any of the candidates or not.

Consider:
- Entity titles/names (including variations, synonyms, abbreviations)
- Entity attributes (if provided)
- Relationship descriptions that provide context about how the entity relates to others

If the current entity refers to the same real world entity as one or more candidates, return that candidates' IDs with confidence 1.0 and match type "exact".
If the current entity has a possible match with one or more candidates, return a list of their IDs with estimated confidence scores and match type "partial".
If the current entity does not match any candidate, return the current entity's ID with confidence 1.0 and match type "none".

## Rules
Include a confidence score between 0 and 1. Use 1 only if you are absolutely sure. 
Use a clear and concise reasoning string to explain your decision. 
Do not include any other text in your response.

# EXAMPLES
## EXAMPLE 1
### INPUT
{{
    "entity": {{
        "id": "0",
        "title": "Founder",
        "attributes": {{
            "type": "PERSON"
        }},
        "relationship_descriptions": [
            "Founder is asked about Competitors.",
            "Founder was proud of creating a Market.",
            "A mistake was heard in a Founder pitch by a Founder."
        ]
    }},
    "candidates": {{
        "1": {{
            "id": "1",
            "title": "Founder",
            "attributes":{{
                "type": "PERSON"
            }},
            "relationship_descriptions": [
                "Like and share with a Founder who needs to hear this. #StartupAdvice",
                "Like and share with a Founder who needs to hear this. #ProductMarketFit"
            ]
        }},
        "2": {{
            "id": "2",
            "title": "Founder pitch",
            "attributes": {{
                "type": "CONCEPT"
            }},
            "relationship_descriptions": [
                "A mistake was heard in a Founder pitch by a Founder.",
                "Founder pitch was about a niche dentist front office software."
            ]
        }}
    }}
}}
### OUTPUT
{{
    "id": "0",
    "canonical_entities": {{
        "1": {{
            "match_type": "exact",
            "confidence": 1.0,
            "reasoning": "The current entity matches the candidate entity 1 perfectly."
        }}
    }}
}}
## EXAMPLE 2
### INPUT
{{
    "entity": {{
        "id": "0",
        "title": "Founders",
        "attributes": {{
            "type": "PERSON_GROUP"
        "relationship_descriptions": [
            "Founders started chasing Compliments.",
            "The hard truth for most Founders is that Theoretical problems rarely pay."
        ]
    }},
    "candidates": {{
        "1": {{
            "id": "1",
            "title": "Startup",
            "attributes": {{
                "type": "ORGANIZATION"
            }},
            "relationship_descriptions": [
                "In Startups, analysis without Action is surrender.",
                "Startups are about solving real problems."
            ]
        }},
        "2": {{
            "id": "2",
            "title": "Founder",
            "attributes": {{
                "type": "PERSON"
            }},
            "relationship_descriptions": [
                "Like and share with a Founder who needs to hear this. #StartupAdvice",
                "Like and share with a Founder who needs to hear this. #ProductMarketFit"
            ]
        }},
        "3": {{
            "id": "3",
            "title": "Core Founding Team",
            "attributes": {{
                "type": "PERSON_GROUP"
            }},
            "relationship_descriptions": [
                "The question is about the Core founding team, which is part of the Founding team.",
                "The Core founding team will be able to see the vision through to a Successful exit."
            ]
        }}
    }}
}}
### OUTPUT
{{
    "id": "0",
    "canonical_entities": {{
        "1": {{
            "match_type": "partial",
            "confidence": 0.8,
            "reasoning": "The current entity matches candidate 2 only partially."
        }},
        "3": {{
            "match_type": "partial",
            "confidence": 0.8,
            "reasoning": "The current entity matches candidate 3 only partially."
        }}
    }}
}}
## EXAMPLE 3
### INPUT
{{
    "entity": {{
        "id": "0",
        "title": "VC firm",
        "attributes": {{
            "type": "ORGANIZATION"
        }},
        "relationship_descriptions": [
            "You want to stack the odds in your favor by directly targeting Decision maker of a VC firm.",
        ]
    }},
    "candidates": {{
        "1": {{
            "title": "Junior Investor",
            "attributes": {{
                "type": "PERSON_GROUP"
            }},
            "relationship_descriptions": [
                "Junior investors include Analysts."
            ]
        }},
        "2": {{
            "title": "General Partner",
            "attributes": {{
                "type": "PERSON_GROUP"
            }},
            "relationship_descriptions": [
                "Many in the VC community may not like this take, which involves talking to General Partners (GP)."
            ]
        }}
    }}
}}
### OUTPUT
{{
    "id": "0",
    "canonical_entities": {{
        "1": {{
            "match_type": "exact",
            "confidence": 1.0,
            "reasoning": "The current entity matches none of the candidates."
        }}
    }}
}}

# REAL DATA
## INPUT
{input_json}
## OUTPUT
"""