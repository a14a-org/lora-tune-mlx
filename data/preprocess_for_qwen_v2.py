#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

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
    def __init__(self):
        self.stats = ProcessingStats()
    
    def format_tool_definition(self, tool_name: str, description: str, parameters: List[Dict]) -> str:
        """Format a tool definition with examples and format rules."""
        formatted = f"<tool_definition name='{tool_name}'>\n"
        formatted += f"  description: {description}\n"
        formatted += "  parameters:\n"
        for param in parameters:
            formatted += f"    - {param['name']} ({param['type']}): {param['description']}\n"
        formatted += "  format_rules:\n"
        formatted += "    - Tool calls must use XML-style tags\n"
        formatted += "    - Parameters must be space-separated key=value pairs\n"
        formatted += "    - String values must be quoted\n"
        formatted += "</tool_definition>\n"
        return formatted

    def format_tool_call(self, name: str, args: Dict) -> str:
        """Format a tool call with consistent parameter formatting."""
        params = []
        for k, v in args.items():
            if isinstance(v, str):
                params.append(f'{k}="{v}"')  # Always quote strings
            elif isinstance(v, (dict, list)):
                params.append(f'{k}={json.dumps(v)}')  # JSON for complex types
            else:
                params.append(f'{k}={v}')  # Raw values for numbers/booleans
        return f"<tool name='{name}'>{' '.join(params)}</tool>"

    def combine_assistant_messages(self, messages: List[Dict]) -> List[Dict]:
        """Combine consecutive assistant messages into single messages."""
        combined = []
        current_assistant_content = []
        
        for msg in messages:
            if msg["role"] == "assistant":
                current_assistant_content.append(msg["content"])
            else:
                # If we have pending assistant content, combine and add it
                if current_assistant_content:
                    combined.append({
                        "role": "assistant",
                        "content": "\n".join(filter(None, current_assistant_content))
                    })
                    current_assistant_content = []
                combined.append(msg)
        
        # Add any remaining assistant content
        if current_assistant_content:
            combined.append({
                "role": "assistant",
                "content": "\n".join(filter(None, current_assistant_content))
            })
        
        return combined

    def convert_example(self, example: Dict) -> Optional[Dict]:
        """Convert a single example to the improved format."""
        try:
            messages = []
            
            # Process system message first
            system_msg = next((msg for msg in example["messages"] if msg["role"] == "system"), None)
            if system_msg:
                # Update system message to include tool definitions in the new format
                tools_content = "You are an AI coding assistant. You help users with coding tasks using the following tools:\n\n"
                
                # Add tool definitions
                for tool in ["list_dir", "read_file", "edit_file", "run_terminal_cmd"]:
                    if tool == "list_dir":
                        tools_content += self.format_tool_definition(
                            tool,
                            "Lists directory contents at a specified path relative to the workspace",
                            [{"name": "relative_workspace_path", "type": "string", "description": "Path to list contents of"}]
                        )
                    elif tool == "read_file":
                        tools_content += self.format_tool_definition(
                            tool,
                            "Reads file contents, with support for both full file and partial reading",
                            [
                                {"name": "relative_workspace_path", "type": "string", "description": "Path to the file"},
                                {"name": "should_read_entire_file", "type": "boolean", "description": "Whether to read entire file"},
                                {"name": "start_line_one_indexed", "type": "integer", "description": "Start line (1-based)"},
                                {"name": "end_line_one_indexed_inclusive", "type": "integer", "description": "End line (1-based)"}
                            ]
                        )
                    elif tool == "edit_file":
                        tools_content += self.format_tool_definition(
                            tool,
                            "Makes code changes to specified files based on instructions",
                            [
                                {"name": "target_file", "type": "string", "description": "File to edit"},
                                {"name": "instructions", "type": "string", "description": "What changes to make"},
                                {"name": "code_edit", "type": "string", "description": "The actual edit content"}
                            ]
                        )
                    elif tool == "run_terminal_cmd":
                        tools_content += self.format_tool_definition(
                            tool,
                            "Executes terminal commands with configurable execution options",
                            [
                                {"name": "command", "type": "string", "description": "Command to execute"},
                                {"name": "is_background", "type": "boolean", "description": "Whether to run in background"},
                                {"name": "require_user_approval", "type": "boolean", "description": "Whether user must approve"}
                            ]
                        )
                
                messages.append({
                    "role": "system",
                    "content": tools_content
                })
            
            # Process remaining messages
            for msg in example["messages"]:
                if msg["role"] == "user":
                    messages.append(msg)
                elif msg["role"] == "assistant":
                    if "function_call" in msg:
                        # Convert function call to tool call format
                        tool_name = msg["function_call"]["name"]
                        args = json.loads(msg["function_call"]["arguments"])
                        tool_call = self.format_tool_call(tool_name, args)
                        
                        # Add natural language response with tool call
                        content = f"I'll help you with that.\n{tool_call}"
                        messages.append({
                            "role": "assistant",
                            "content": content
                        })
                        
                        # Track tool usage
                        self.stats.tool_usage[tool_name] = self.stats.tool_usage.get(tool_name, 0) + 1
                    else:
                        messages.append(msg)
                elif msg["role"] == "function":
                    # Convert function response to tool response format
                    response_data = json.loads(msg["content"])
                    formatted_response = f"<tool_response name='{msg['name']}'>{json.dumps(response_data)}</tool_response>"
                    messages.append({
                        "role": "system",
                        "content": formatted_response
                    })
            
            # Update statistics
            self.stats.total_examples += 1
            self.stats.successful_conversions += 1
            
            return {"messages": messages}
            
        except Exception as e:
            logger.error(f"Error converting example: {e}")
            self.stats.failed_conversions += 1
            return None

    def process_dataset(self, input_file: Path, output_file: Path) -> None:
        """Process the entire dataset."""
        logger.info(f"Processing {input_file} -> {output_file}")
        
        try:
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
            
            # Process examples
            processed = []
            for example in examples:
                result = self.convert_example(example)
                if result:
                    processed.append(result)
            
            # Write output
            with open(output_file, 'w') as f:
                for example in processed:
                    f.write(json.dumps(example) + '\n')
            
            logger.info(f"Processed {len(processed)} examples")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to improved Qwen format')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Input directory containing dataset files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for processed files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessor
        preprocessor = QwenPreprocessor()
        
        # Find input files
        train_files = sorted(input_dir.glob("train_*.json"))
        valid_files = sorted(input_dir.glob("validation_*.json"))
        
        if not train_files or not valid_files:
            raise FileNotFoundError("No dataset files found")
        
        # Use most recent files
        train_file = train_files[-1]
        valid_file = valid_files[-1]
        
        # Process datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_output = output_dir / f"train_{timestamp}.jsonl"
        valid_output = output_dir / f"valid_{timestamp}.jsonl"
        stats_output = output_dir / f"processing_stats_{timestamp}.json"
        
        preprocessor.process_dataset(train_file, train_output)
        preprocessor.process_dataset(valid_file, valid_output)
        
        # Save processing statistics
        with open(stats_output, 'w') as f:
            json.dump(preprocessor.stats.to_dict(), f, indent=2)
        
        logger.info(f"Processing complete. Files saved to {output_dir}")
        logger.info(f"Statistics saved to {stats_output}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main() 