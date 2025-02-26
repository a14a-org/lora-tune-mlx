"""Utility functions for formatting tool calls and responses."""

import json
import random
from typing import Dict, Any, List, Optional

def generate_tool_call(tool_name: str, args: Dict[str, Any]) -> str:
    """Generate a tool call using Qwen's native tokens."""
    args_str = " ".join([f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in args.items()])
    return f"<tool_call>{tool_name} {args_str}</tool_call>"

def generate_tool_response(tool_name: str, response_data: Dict[str, Any]) -> str:
    """Generate a tool response."""
    return json.dumps(response_data)

def format_conversation(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format a conversation into the expected structure."""
    return {"messages": messages}

def select_random_template(templates: List[str], **kwargs) -> str:
    """Select and format a random template."""
    template = random.choice(templates)
    return template.format(**kwargs)

def create_message(role: str, content: str, function_call: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a message with the given role and content."""
    message = {
        "role": role,
        "content": content
    }
    if function_call:
        message["function_call"] = function_call
    return message 