"""Base class for conversation generators."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from data.utils.formatting import (
    generate_tool_call,
    generate_tool_response,
    format_conversation,
    create_message,
    select_random_template
)

class BaseGenerator(ABC):
    """Base class for generating conversations."""
    
    def __init__(self):
        """Initialize the generator."""
        self.messages: List[Dict[str, Any]] = []
        
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(create_message("user", content))
        
    def add_assistant_message(self, content: str, tool_call: Dict[str, Any] = None) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(create_message("assistant", content, tool_call))
        
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.messages.append(create_message("system", content))
        
    def add_tool_call(self, tool_name: str, args: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """Add a tool call and its response to the conversation."""
        tool_call = generate_tool_call(tool_name, args)
        self.add_assistant_message("", {"name": tool_name, "arguments": args})
        self.add_system_message(generate_tool_response(tool_name, response_data))
        
    def get_conversation(self) -> Dict[str, Any]:
        """Get the formatted conversation."""
        return format_conversation(self.messages)
        
    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        """Generate a conversation. Must be implemented by subclasses."""
        pass

class BaseGenerator:
    """Base class for all scenario generators."""
    
    def generate(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a scenario.
        
        Returns:
            Dictionary containing the generated scenario
        """
        raise NotImplementedError("Subclasses must implement generate()") 