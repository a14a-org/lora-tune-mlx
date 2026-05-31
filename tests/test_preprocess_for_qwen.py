"""Unit tests for the pure formatting/parsing helpers in
``data/preprocess_for_qwen.py``.

These functions transform tool-calling training data into the textual format
the model is fine-tuned on, so their exact output is load-bearing. The tests
assert on the concrete strings and parsed structures the rest of the pipeline
relies on.
"""


def test_format_tool_call_quotes_strings(preprocess):
    result = preprocess.format_tool_call("control_lights", {"room": "kitchen", "action": "on"})
    assert result == '<tool name=\'control_lights\'>room="kitchen" action="on"</tool>'


def test_format_tool_call_leaves_numbers_unquoted(preprocess):
    result = preprocess.format_tool_call("set_thermostat", {"temperature": 22})
    assert result == "<tool name='set_thermostat'>temperature=22</tool>"


def test_format_tool_call_booleans_are_raw(preprocess):
    result = preprocess.format_tool_call("toggle", {"enabled": True})
    # Python's str(True) -> "True"; the helper emits the raw value for non-strings.
    assert result == "<tool name='toggle'>enabled=True</tool>"


def test_format_tool_call_json_encodes_complex_values(preprocess):
    result = preprocess.format_tool_call("schedule", {"days": ["mon", "tue"]})
    assert result == '<tool name=\'schedule\'>days=["mon", "tue"]</tool>'


def test_format_tool_call_mixed_argument_types(preprocess):
    result = preprocess.format_tool_call("set_thermostat", {"temperature": 20, "unit": "C"})
    assert result == "<tool name='set_thermostat'>temperature=20 unit=\"C\"</tool>"


def test_format_tool_definition_includes_name_and_params(preprocess):
    out = preprocess.format_tool_definition(
        "get_weather",
        "Get the current weather for a location",
        [{"name": "location", "type": "string", "description": "City name"}],
    )
    assert "<tool_definition name='get_weather'>" in out
    assert "description: Get the current weather for a location" in out
    assert "- location (string): City name" in out
    # Each definition block is terminated with the closing tag.
    assert out.rstrip().endswith("</tool_definition>")


def test_parse_tool_info_extracts_prefixed_tool(preprocess):
    # The tool name is taken from the first token of the header line; the
    # description is taken from the next free-text line, and parameters from the
    # bulleted ("*") lines under "Parameters:".
    content = (
        "control_lights\n"
        "Controls the lights in a room\n"
        "Parameters:\n"
        "* room (string): The room to control\n"
        "* brightness (int): Brightness percentage\n"
    )
    tools = preprocess.parse_tool_info(content)
    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "control_lights"
    assert tool["description"] == "Controls the lights in a room"
    assert tool["parameters"] == [
        {"name": "room", "type": "string", "description": "The room to control"},
        {"name": "brightness", "type": "int", "description": "Brightness percentage"},
    ]


def test_parse_tool_info_handles_no_tools(preprocess):
    assert preprocess.parse_tool_info("Just a plain sentence with no tools.") == []
