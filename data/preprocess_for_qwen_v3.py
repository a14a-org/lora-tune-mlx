#!/usr/bin/env python3

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for data processing"""
    total_examples: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    tool_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.tool_usage is None:
            self.tool_usage = {}
    
    def to_dict(self) -> Dict:
        return {
            "total_examples": self.total_examples,
            "successful_conversions": self.successful_conversions,
            "failed_conversions": self.failed_conversions,
            "tool_usage": self.tool_usage
        }

class QwenPreprocessor:
    """Preprocessor for converting conversations to Qwen format with native tokens."""
    
    def __init__(self):
        self.reset_stats()
        # Define available tools
        self.tools = {
            "list_dir": {
                "description": "Lists directory contents at a specified path relative to the workspace",
                "parameters": [
                    {"name": "relative_workspace_path", "type": "string", "description": "Path to list contents of"}
                ]
            },
            "read_file": {
                "description": "Reads file contents, with support for both full file and partial reading",
                "parameters": [
                    {"name": "relative_workspace_path", "type": "string", "description": "Path to the file"},
                    {"name": "should_read_entire_file", "type": "boolean", "description": "Whether to read entire file"},
                    {"name": "start_line_one_indexed", "type": "integer", "description": "Start line (1-based)"},
                    {"name": "end_line_one_indexed_inclusive", "type": "integer", "description": "End line (1-based)"}
                ]
            },
            "edit_file": {
                "description": "Makes code changes to specified files based on instructions",
                "parameters": [
                    {"name": "target_file", "type": "string", "description": "File to edit"},
                    {"name": "instructions", "type": "string", "description": "What changes to make"},
                    {"name": "code_edit", "type": "string", "description": "The actual edit content"}
                ]
            },
            "run_terminal_cmd": {
                "description": "Executes terminal commands with configurable execution options",
                "parameters": [
                    {"name": "command", "type": "string", "description": "Command to execute"},
                    {"name": "is_background", "type": "boolean", "description": "Whether to run in background"},
                    {"name": "require_user_approval", "type": "boolean", "description": "Whether user must approve"}
                ]
            }
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = ProcessingStats()
    
    def format_tool_definition(self, tool_name: str, tool_info: Dict) -> str:
        """Format a tool definition with examples and format rules."""
        formatted = f"<tool_definition name='{tool_name}'>\n"
        formatted += f"  description: {tool_info['description']}\n"
        formatted += "  parameters:\n"
        for param in tool_info['parameters']:
            formatted += f"    - {param['name']} ({param['type']}): {param['description']}\n"
        formatted += "  format_rules:\n"
        formatted += "    - Tool calls must use XML-style tags\n"
        formatted += "    - Parameters must be space-separated key=value pairs\n"
        formatted += "    - String values must be quoted\n"
        formatted += "</tool_definition>\n"
        return formatted
    
    def format_tool_call(self, name: str, args: Dict) -> str:
        """Format a tool call using Qwen's native tokens."""
        args_str = " ".join([f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in args.items()])
        return f"<tool_call>{name} {args_str}</tool_call>"
    
    def format_tool_response(self, name: str, response_data: Dict) -> str:
        """Format a tool response."""
        return f"<|im_start|>system\n{json.dumps(response_data)}\n<|im_end|>"
    
    def get_system_prompt(self) -> str:
        """Generate the system prompt with tool definitions."""
        prompt = "You are a powerful agentic AI coding assistant. You help users with coding tasks using the following tools:\n\n"
        
        # Add tool definitions
        for tool_name, tool_info in self.tools.items():
            prompt += self.format_tool_definition(tool_name, tool_info)
        
        # Add format rules and examples
        prompt += "\nFormat Rules:\n"
        prompt += "1. Always use XML-style tags for tool calls\n"
        prompt += "2. Quote string parameter values\n"
        prompt += "3. Use space-separated key=value pairs for parameters\n"
        
        return prompt
    
    def convert_example(self, example: Dict) -> Optional[Dict]:
        """Convert a single example to Qwen format."""
        try:
            messages = []
            
            # Add system message with tool definitions first
            messages.append({
                "role": "system",
                "content": self.get_system_prompt()
            })
            
            # Process the rest of the messages
            for msg in example["messages"]:
                if msg["role"] == "user":
                    messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant":
                    content = msg.get("content", "")
                    if "function_call" in msg:
                        tool_name = msg["function_call"]["name"]
                        # Handle arguments that might be string or dict
                        args = msg["function_call"]["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        tool_call = self.format_tool_call(tool_name, args)
                        content = f"{content}\n{tool_call}" if content else tool_call
                        self.stats.tool_usage[tool_name] = self.stats.tool_usage.get(tool_name, 0) + 1
                    
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
                elif msg["role"] == "function":
                    # Handle content that might be string or dict
                    response_data = msg["content"]
                    if isinstance(response_data, str):
                        response_data = json.loads(response_data)
                    messages.append({
                        "role": "system",
                        "content": self.format_tool_response(msg["name"], response_data)
                    })
            
            self.stats.successful_conversions += 1
            return {"messages": messages}
            
        except Exception as e:
            logger.error(f"Error converting example: {str(e)}")
            self.stats.failed_conversions += 1
            return None

    def process_dataset(self, input_file: Path, output_file: Path) -> None:
        """Process the entire dataset."""
        logger.info(f"Processing {input_file} -> {output_file}")
        
        try:
            # Reset statistics for this dataset
            self.reset_stats()
            
            # Read input data
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Handle nested data structure
            if isinstance(data, dict) and 'data' in data:
                examples = data['data']
            elif isinstance(data, list):
                examples = data
            else:
                examples = [data]
            
            self.stats.total_examples = len(examples)
            
            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Process examples
            with open(output_file, 'w') as f:
                for example in examples:
                    result = self.convert_example(example)
                    if result:
                        json_str = json.dumps(result)
                        f.write(json_str + '\n')
            
            # Save statistics
            stats_file = output_file.parent / f"processing_stats_{output_file.stem}.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            
            logger.info(f"Processed {self.stats.successful_conversions} examples successfully")
            logger.info(f"Failed to process {self.stats.failed_conversions} examples")
            logger.info(f"Tool usage statistics: {self.stats.tool_usage}")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess conversation data for Qwen format')
    parser.add_argument('input', type=Path, help='Input JSON file')
    parser.add_argument('output', type=Path, help='Output JSONL file')
    args = parser.parse_args()
    
    preprocessor = QwenPreprocessor()
    preprocessor.process_dataset(args.input, args.output)

if __name__ == "__main__":
    main() 