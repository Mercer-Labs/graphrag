# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Test NodeReferences."""

from graphrag.index.operations.extract_graph.node_references import NodeReferences


class TestEncodeNodeReferencesToLlmOutput:
    """Test encode_node_references_to_llm_output method."""

    def test_basic_replacement(self):
        """Test basic replacement of placeholders with node identifiers."""
        text = "The relationship between $##1 and $##2 is important."
        source_id = "entity_123"
        target_id = "entity_456"
        
        result = NodeReferences.encode_node_references_in_llm_output(
            text, source_id, target_id
        )
        
        expected = f"The relationship between &#{{entity_123}} and &#{{entity_456}} is important."
        assert result == expected

    def test_source_placeholder_only(self):
        """Test replacement when only source placeholder is present."""
        text = "This is about $##1."
        source_id = "source_node"
        target_id = "target_node"
        
        result = NodeReferences.encode_node_references_in_llm_output(
            text, source_id, target_id
        )
        
        assert "&#{source_node}" in result
        assert "$##1" not in result
        assert "$##2" not in result

    def test_target_placeholder_only(self):
        """Test replacement when only target placeholder is present."""
        text = "This is about $##2."
        source_id = "source_node"
        target_id = "target_node"
        
        result = NodeReferences.encode_node_references_in_llm_output(
            text, source_id, target_id
        )
        
        assert "&#{target_node}" in result
        assert "$##2" not in result

    def test_multiple_placeholders(self):
        """Test replacement of multiple occurrences of placeholders."""
        text = "$##1 and $##1 are related. Also $##2 and $##2."
        source_id = "source"
        target_id = "target"
        
        result = NodeReferences.encode_node_references_in_llm_output(
            text, source_id, target_id
        )
        
        assert result.count("&#{source}") == 2
        assert result.count("&#{target}") == 2
        assert "$##1" not in result
        assert "$##2" not in result

    def test_escape_existing_node_identifiers(self):
        """Test that existing node identifiers are escaped."""
        text = "This has &#{existing} and $##1."
        source_id = "new_node"
        target_id = "target"
        
        result = NodeReferences.encode_node_references_in_llm_output(
            text, source_id, target_id
        )
        
        # Existing identifier should be escaped
        assert "&&##{existing}" in result
        # New placeholder should be replaced
        assert "&#{new_node}" in result
        assert "$##1" not in result

    def test_empty_text(self):
        """Test with empty text."""
        result = NodeReferences.encode_node_references_in_llm_output(
            "", "source", "target"
        )
        assert result == ""

    def test_no_placeholders(self):
        """Test text without placeholders."""
        text = "This text has no placeholders."
        result = NodeReferences.encode_node_references_in_llm_output(
            text, "source", "target"
        )
        assert result == text

    def test_special_characters_in_ids(self):
        """Test with special characters in node IDs."""
        text = "$##1 and $##2"
        source_id = "node-with-dashes_123"
        target_id = "node.with.dots"
        
        result = NodeReferences.encode_node_references_in_llm_output(
            text, source_id, target_id
        )
        
        assert "&#{node-with-dashes_123}" in result
        assert "&#{node.with.dots}" in result

    def test_unicode_characters_in_ids(self):
        """Test with unicode characters in node IDs."""
        text = "$##1"
        source_id = "node_测试"
        target_id = "target"
        
        result = NodeReferences.encode_node_references_in_llm_output(
            text, source_id, target_id
        )
        
        assert "&#{node_测试}" in result


class TestHydrateNodeReferences:
    """Test hydrate_node_references method."""

    def test_basic_hydration(self):
        """Test basic replacement of node references with titles."""
        text = "The relationship between &#{entity_123} and &#{entity_456} is important."
        node_map = {
            "entity_123": {"title": "Source Entity"},
            "entity_456": {"title": "Target Entity"},
        }
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "The relationship between Source Entity and Target Entity is important."

    def test_single_reference(self):
        """Test hydration with a single reference."""
        text = "This is about &#{node1}."
        node_map = {"node1": {"title": "Node One"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "This is about Node One."

    def test_multiple_references(self):
        """Test hydration with multiple references."""
        text = "&#{a} and &#{b} and &#{c}"
        node_map = {
            "a": {"title": "A"},
            "b": {"title": "B"},
            "c": {"title": "C"},
        }
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "A and B and C"

    def test_missing_node_in_map(self):
        """Test that missing nodes are left unchanged."""
        text = "This is about &#{missing} and &#{found}."
        node_map = {"found": {"title": "Found Node"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert "&#{missing}" in result
        assert "Found Node" in result

    def test_custom_title_key(self):
        """Test with a custom title key."""
        text = "This is &#{node1}."
        node_map = {"node1": {"name": "Custom Name"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map, title_key="name")
        
        assert result == "This is Custom Name."

    def test_unescape_escaped_identifiers(self):
        """Test that escaped identifiers are unescaped."""
        text = "This has &&##{escaped} and &#{normal}."
        node_map = {"normal": {"title": "Normal Node"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        # Escaped identifier should be unescaped
        assert "&#{escaped}" in result
        # Normal reference should be hydrated
        assert "Normal Node" in result

    def test_empty_text(self):
        """Test with empty text."""
        result = NodeReferences.hydrate_node_references("", {})
        assert result == ""

    def test_no_references(self):
        """Test text without node references."""
        text = "This text has no references."
        result = NodeReferences.hydrate_node_references(text, {})
        assert result == text

    def test_empty_node_map(self):
        """Test with empty node map."""
        text = "This has &#{node1}."
        result = NodeReferences.hydrate_node_references(text, {})
        # Reference should remain unchanged
        assert "&#{node1}" in result

    def test_special_characters_in_ids(self):
        """Test with special characters in node IDs."""
        text = "This is &#{node-123}."
        node_map = {"node-123": {"title": "Special Node"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "This is Special Node."

    def test_nested_braces_in_text(self):
        """Test that text with nested braces doesn't break the regex."""
        text = "This has {some} braces and &#{node1}."
        node_map = {"node1": {"title": "Node"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert "{some}" in result
        assert "Node" in result
        assert "&#{node1}" not in result

    def test_reference_at_start(self):
        """Test reference at the start of text."""
        text = "&#{node1} is important."
        node_map = {"node1": {"title": "Node"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "Node is important."

    def test_reference_at_end(self):
        """Test reference at the end of text."""
        text = "This is about &#{node1}"
        node_map = {"node1": {"title": "Node"}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "This is about Node"

    def test_consecutive_references(self):
        """Test consecutive references without spaces."""
        text = "&#{a}&#{b}"
        node_map = {
            "a": {"title": "A"},
            "b": {"title": "B"},
        }
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "AB"

    def test_node_with_empty_title(self):
        """Test node with empty title."""
        text = "This is &#{node1}."
        node_map = {"node1": {"title": ""}}
        
        result = NodeReferences.hydrate_node_references(text, node_map)
        
        assert result == "This is ."


class TestRoundTrip:
    """Test round-trip encoding and hydration."""

    def test_round_trip_basic(self):
        """Test encoding then hydrating produces original text with titles."""
        original_text = "The relationship between $##1 and $##2 is important."
        source_id = "entity_123"
        target_id = "entity_456"
        node_map = {
            "entity_123": {"title": "Source Entity"},
            "entity_456": {"title": "Target Entity"},
        }
        
        # Encode
        encoded = NodeReferences.encode_node_references_in_llm_output(
            original_text, source_id, target_id
        )
        
        # Hydrate
        hydrated = NodeReferences.hydrate_node_references(encoded, node_map)
        
        assert hydrated == "The relationship between Source Entity and Target Entity is important."

    def test_round_trip_with_existing_identifiers(self):
        """Test round-trip with existing identifiers that need escaping."""
        original_text = "This has &#{existing} and $##1."
        source_id = "new_node"
        target_id = "target"
        node_map = {
            "existing": {"title": "Existing Node"},
            "new_node": {"title": "New Node"},
            "target": {"title": "Target Node"},
        }
        
        # Encode
        encoded = NodeReferences.encode_node_references_in_llm_output(
            original_text, source_id, target_id
        )
        
        # Hydrate
        hydrated = NodeReferences.hydrate_node_references(encoded, node_map)
        
        # Existing identifier should be preserved
        assert "&#{existing}" in hydrated
        # New placeholder should be replaced
        assert "New Node" in hydrated

