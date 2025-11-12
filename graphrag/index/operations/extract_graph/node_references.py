# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module for managing references to objects in strings."""

import re
from typing import Any, Mapping


# simple handling of placeholders in the description. TODO SUBU make this better.
class NodeReferences:
    SOURCE_PLACEHOLDER = "$##1"
    TARGET_PLACEHOLDER = "$##2"

    NODE_IDENTIFIER_PREFIX = "&#"
    NODE_IDENTIFIER_ENCODING_FORMAT = NODE_IDENTIFIER_PREFIX + "{{{id}}}"
    NODE_IDENTIFIER_PREFIX_ESCAPE = "&&##"

    """Manages references to nodes in text descriptions
    Extract_graph prompt is tuned to output references to nodes using placeholders.
    This class provides functionality to:
    - Replace placeholders with the references to node identifiers
    - Replace references to node identifiers with the titles
    """

    @staticmethod
    def cleanup_placeholders_in_text(text: str) -> str:
        """Cleanup placeholders in text.
        
        Args:
            text: Text containing placeholders to be cleaned up
        Returns:
            Text with placeholders cleaned up
        """
        return text.replace(NodeReferences.SOURCE_PLACEHOLDER, "").replace(NodeReferences.TARGET_PLACEHOLDER, "")

    @staticmethod
    def encode_node_references_in_llm_output(
        text: str, source_id: str, target_id: str
    ) -> str:
        """Encode node references to LLM output that uses placeholders.
        
        Args:
            text: LLM output (description) Text containing placeholders ($##1 and $##2) to be encoded
            source_id: The identifier for the source node
            target_id: The identifier for the target node
            
        Returns:
            Text with placeholders replaced by identifier references in the format ${id}
        """
        # Replace {id} in the format template with the actual identifier
        source_reference = NodeReferences.NODE_IDENTIFIER_ENCODING_FORMAT.format(
            id=source_id
        )
        target_reference = NodeReferences.NODE_IDENTIFIER_ENCODING_FORMAT.format(
            id=target_id
        )
        # escape any existing node identifiers in the text
        text = text.replace(NodeReferences.NODE_IDENTIFIER_PREFIX, NodeReferences.NODE_IDENTIFIER_PREFIX_ESCAPE)

        text = text.replace(NodeReferences.SOURCE_PLACEHOLDER, source_reference)
        text = text.replace(NodeReferences.TARGET_PLACEHOLDER, target_reference)
        return text

    @staticmethod
    def hydrate_node_references(
        text: str, node_map: Mapping[str, Any], title_key: str = "title"
    ) -> str:
        """Hydrate node references with node titles to get raw text.
        
        Args:
            text: Text containing node identifier references in the format &#{id}
            node_map: Mapping from node identifiers to nodes
            Optional title_key: Key to access the title in the node map
        Returns:
            Text with node references replaced by corresponding titles
        """
        # Pattern to match &#{id} where id can be any string
        pattern = re.compile(r"&#\{([^}]+)\}")
        
        def replace_match(match: re.Match[str]) -> str:
            node_id = match.group(1)
            node = node_map.get(node_id, None)
            if node:
                return node[title_key]
            return match.group(0)
        
        text = pattern.sub(replace_match, text)

        # unescape any existing node identifiers in the text
        text = text.replace(NodeReferences.NODE_IDENTIFIER_PREFIX_ESCAPE, NodeReferences.NODE_IDENTIFIER_PREFIX)
        return text

