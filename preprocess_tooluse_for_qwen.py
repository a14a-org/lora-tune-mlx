#!/usr/bin/env python3
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for the preprocessing operation"""
    total_examples: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    scenario_types: Dict[str, int] = None
    tool_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.scenario_types is None:
            self.scenario_types = defaultdict(int)
        if self.tool_usage is None:
            self.tool_usage = defaultdict(int)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "successful_conversions": self.successful_conversions,
            "failed_conversions": self.failed_conversions,
            "scenario_types": dict(self.scenario_types),
            "tool_usage": dict(self.tool_usage)
        }

class QwenToolPreprocessor:
    """Preprocesses tool usage datasets into Qwen-compatible format"""
    
    def __init__(self):
        # Tool definitions based on the available tools in the dataset
        self.tools = {
            "list_dir": {
                "name": "list_dir",
                "description": "Lists directory contents at a specified path relative to the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_workspace_path": {
                            "type": "string",
                            "description": "Path to list contents of"
                        }
                    },
                    "required": ["relative_workspace_path"]
                }
            },
            "read_file": {
                "name": "read_file",
                "description": "Reads file contents, with support for both full file and partial reading",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_workspace_path": {
                            "type": "string",
                            "description": "Path to the file"
                        },
                        "should_read_entire_file": {
                            "type": "boolean",
                            "description": "Whether to read entire file"
                        },
                        "start_line_one_indexed": {
                            "type": "integer",
                            "description": "Start line (1-based)"
                        },
                        "end_line_one_indexed_inclusive": {
                            "type": "integer",
                            "description": "End line (1-based)"
                        }
                    },
                    "required": ["relative_workspace_path", "should_read_entire_file", 
                               "start_line_one_indexed", "end_line_one_indexed_inclusive"]
                }
            },
            "edit_file": {
                "name": "edit_file",
                "description": "Makes code changes to specified files based on instructions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "File to edit"
                        },
                        "instructions": {
                            "type": "string",
                            "description": "What changes to make"
                        },
                        "code_edit": {
                            "type": "string",
                            "description": "The actual edit content"
                        }
                    },
                    "required": ["target_file", "instructions", "code_edit"]
                }
            },
            "run_terminal_cmd": {
                "name": "run_terminal_cmd",
                "description": "Executes terminal commands with configurable execution options",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute"
                        },
                        "is_background": {
                            "type": "boolean",
                            "description": "Whether to run in background"
                        },
                        "require_user_approval": {
                            "type": "boolean",
                            "description": "Whether user must approve"
                        }
                    },
                    "required": ["command", "is_background", "require_user_approval"]
                }
            }
        }
        
        self.stats = ProcessingStats()
    
    def format_tool_definition(self, tool_name: str, tool_info: Dict) -> str:
        """Format a tool definition in Qwen's expected format"""
        formatted = f"<tool_definition name='{tool_name}'>\n"
        formatted += f"  description: {tool_info['description']}\n"
        formatted += "  parameters:\n"
        
        for param_name, param_info in tool_info['parameters']['properties'].items():
            required = param_name in tool_info['parameters'].get('required', [])
            formatted += f"    - {param_name} ({param_info['type']}{', required' if required else ''}): {param_info['description']}\n"
        
        formatted += "</tool_definition>\n"
        return formatted
    
    def format_tool_call(self, name: str, args: Dict) -> str:
        """Format a tool call in Qwen's XML-style format"""
        params = []
        for k, v in args.items():
            if isinstance(v, str):
                params.append(f'{k}="{v}"')
            elif isinstance(v, (dict, list)):
                params.append(f'{k}={json.dumps(v)}')
            else:
                params.append(f'{k}={v}')
        return f"<tool name='{name}'>{' '.join(params)}</tool>"
    
    def format_tool_response(self, name: str, response_data: Dict) -> str:
        """Format a tool response in Qwen's XML-style format"""
        return f"<tool_response name='{name}'>{json.dumps(response_data)}</tool_response>"
    
    def convert_scenario_to_qwen(self, scenario: Dict) -> Optional[Dict]:
        """Convert a single scenario to Qwen format"""
        try:
            messages = []
            metadata = scenario.get('metadata', {})
            
            # Add system message with tool definitions
            system_msg = "You are an AI coding assistant. You help users with coding tasks using the following tools:\n\n"
            for tool_name, tool_info in self.tools.items():
                system_msg += self.format_tool_definition(tool_name, tool_info)
            
            messages.append({
                "role": "system",
                "content": system_msg
            })
            
            # Process conversation messages
            for msg in scenario['messages']:
                if msg['role'] == 'user':
                    messages.append({
                        "role": "user",
                        "content": msg['content']
                    })
                elif msg['role'] == 'assistant':
                    content = msg.get('content', '')
                    if 'function_call' in msg:
                        tool_name = msg['function_call']['name']
                        args = json.loads(msg['function_call']['arguments'])
                        tool_call = self.format_tool_call(tool_name, args)
                        content = f"{content}\n{tool_call}" if content else tool_call
                        
                        # Update tool usage statistics
                        self.stats.tool_usage[tool_name] += 1
                    
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
                elif msg['role'] == 'function':
                    response_data = json.loads(msg['content'])
                    messages.append({
                        "role": "system",
                        "content": self.format_tool_response(msg['name'], response_data)
                    })
            
            # Update statistics
            self.stats.total_examples += 1
            self.stats.successful_conversions += 1
            if metadata.get('scenario_type'):
                self.stats.scenario_types[metadata['scenario_type']] += 1
            
            return {"messages": messages}
            
        except Exception as e:
            logger.error(f"Failed to convert scenario: {e}", exc_info=True)
            self.stats.failed_conversions += 1
            return None
    
    def process_dataset(self, input_path: Path, output_path: Path):
        """Process a dataset file"""
        logger.info(f"Processing {input_path} -> {output_path}")
        
        try:
            # Read input data
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert scenarios
            processed_examples = []
            for example in data.get('data', []):
                processed = self.convert_scenario_to_qwen(example)
                if processed:
                    processed_examples.append(processed)
            
            # Write output
            with open(output_path, 'w') as f:
                for example in processed_examples:
                    f.write(json.dumps(example) + '\n')
            
            logger.info(f"Successfully processed {len(processed_examples)} examples")
            
        except Exception as e:
            logger.error(f"Failed to process dataset: {e}", exc_info=True)
            raise

def main():
    parser = argparse.ArgumentParser(description='Convert tool usage dataset to Qwen format')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Input directory containing dataset files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for Qwen format files')
    parser.add_argument('--timestamp', type=str,
                      help='Specific timestamp to process (format: YYYYMMDD_HHMMSS)')
    parser.add_argument('--validate-only', action='store_true',
                      help='Only validate the input files without processing')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessor
        preprocessor = QwenToolPreprocessor()
        
        # Find input files
        if args.timestamp:
            train_file = input_dir / f"train_{args.timestamp}.json"
            valid_file = input_dir / f"validation_{args.timestamp}.json"
            stats_file = input_dir / f"statistics_{args.timestamp}.json"
        else:
            # Find most recent files if no timestamp specified
            train_files = sorted(input_dir.glob("train_*.json"))
            valid_files = sorted(input_dir.glob("validation_*.json"))
            stats_files = sorted(input_dir.glob("statistics_*.json"))
            
            if not train_files or not valid_files:
                raise FileNotFoundError("No dataset files found")
            
            train_file = train_files[-1]
            valid_file = valid_files[-1]
            stats_file = stats_files[-1] if stats_files else None
        
        if args.validate_only:
            # Validate input files
            logger.info("Validating input files...")
            for file in [train_file, valid_file]:
                with open(file, 'r') as f:
                    json.load(f)  # Validate JSON format
            logger.info("Input files validated successfully")
            return
        
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
        
        logger.info(f"Processing complete. Statistics saved to {stats_output}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 