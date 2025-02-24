#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_tool_definition(tool_name: str, description: str, parameters: List[Dict]) -> str:
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
    formatted += "  examples:\n"
    
    # Add example calls based on tool type
    if tool_name == "control_lights":
        formatted += """    - User: Turn on the living room lights
      Assistant: I'll help you with that.
      <tool name='control_lights'>room="living room" action="on"</tool>
      System: <tool_response name='control_lights'>{"status":"success","message":"Lights turned on"}</tool_response>
      Assistant: The living room lights have been turned on.
      
    - User: Dim the bedroom lights to 50%
      Assistant: I'll dim the bedroom lights for you.
      <tool name='control_lights'>room="bedroom" action="dim" brightness=50</tool>
      System: <tool_response name='control_lights'>{"status":"success","message":"Lights dimmed"}</tool_response>
      Assistant: I've dimmed the bedroom lights to 50% brightness."""
    elif tool_name == "get_weather":
        formatted += """    - User: What's the weather in Amsterdam?
      Assistant: I'll check the weather in Amsterdam for you.
      <tool name='get_weather'>location="Amsterdam"</tool>
      System: <tool_response name='get_weather'>{"temperature":18,"conditions":"sunny"}</tool_response>
      Assistant: In Amsterdam, it's currently 18°C and sunny."""
    elif tool_name == "set_thermostat":
        formatted += """    - User: Set the temperature to 22 degrees Celsius
      Assistant: I'll set the thermostat to 22°C.
      <tool name='set_thermostat'>temperature=22 unit="C"</tool>
      System: <tool_response name='set_thermostat'>{"status":"success","set_temp":22,"unit":"C"}</tool_response>
      Assistant: The thermostat has been set to 22°C."""
    elif tool_name == "set_thermostat_delta":
        formatted += """    - User: Make it 2 degrees warmer
      Assistant: I'll increase the temperature by 2 degrees.
      <tool name='set_thermostat_delta'>delta=2</tool>
      System: <tool_response name='set_thermostat_delta'>{"status":"success","delta":2,"new_temp":24}</tool_response>
      Assistant: I've increased the temperature by 2 degrees."""
    
    formatted += "\n</tool_definition>\n\n"
    return formatted

def format_tool_call(name: str, args: Dict) -> str:
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

def add_conversation_examples() -> str:
    """Add complete conversation examples to the system message."""
    return """
Example multi-turn conversations:

1. Light control with feedback:
User: Can you turn on the kitchen lights?
Assistant: I'll help you turn on the kitchen lights.
<tool name='control_lights'>room="kitchen" action="on"</tool>
System: <tool_response name='control_lights'>{"status":"success"}</tool_response>
Assistant: The kitchen lights have been turned on successfully.

2. Weather and temperature adjustment:
User: What's the weather like in London?
Assistant: I'll check the weather in London for you.
<tool name='get_weather'>location="London"</tool>
System: <tool_response name='get_weather'>{"temperature":15,"conditions":"cloudy"}</tool_response>
Assistant: In London, it's currently 15°C and cloudy. Would you like me to adjust the thermostat to match the outdoor temperature?
User: Yes, please set it to 20 degrees
Assistant: I'll set the thermostat to 20°C for you.
<tool name='set_thermostat'>temperature=20 unit="C"</tool>
System: <tool_response name='set_thermostat'>{"status":"success","set_temp":20}</tool_response>
Assistant: I've set the thermostat to 20°C for your comfort.
"""

def parse_tool_info(content: str) -> List[Dict]:
    """Parse tool information from the system message content."""
    tools = []
    lines = content.split('\n')
    current_tool = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for tool names
        if (line[0].isdigit() and ". " in line) or any(line.startswith(prefix) for prefix in ["get_", "control_", "set_"]):
            # Extract tool name
            if line[0].isdigit():
                tool_name = line.split(" ")[1].strip("[]")
            else:
                tool_name = line.split(" ")[0].strip("[]")
            
            if tool_name and not tool_name.startswith("You"):
                current_tool = {
                    "name": tool_name,
                    "description": "",
                    "parameters": []
                }
                tools.append(current_tool)
        
        # Parse description and parameters
        elif current_tool is not None:
            if "Parameters:" in line:
                continue
            elif line.startswith("*"):
                param_info = line.strip("* ")
                param_parts = param_info.split(":")
                if len(param_parts) >= 2:
                    param_name = param_parts[0].strip()
                    param_desc = ":".join(param_parts[1:]).strip()
                    param_type = "string"  # Default type
                    
                    # Extract type information
                    if "(" in param_name and ")" in param_name:
                        type_start = param_name.find("(")
                        type_end = param_name.find(")")
                        param_type = param_name[type_start + 1:type_end]
                        param_name = param_name[:type_start].strip()
                    
                    current_tool["parameters"].append({
                        "name": param_name,
                        "type": param_type,
                        "description": param_desc
                    })
            elif current_tool["description"] == "":
                current_tool["description"] = line
    
    return tools

def convert_to_mlx_format_v2(example: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Enhanced version of the conversion function with improved formatting."""
    mlx_messages = []
    
    # Process messages in order, ensuring system message comes first
    system_message = None
    other_messages = []
    
    for msg in example["messages"]:
        if msg["role"] == "system":
            if "You are a home automation assistant" in msg["content"]:
                # Parse tool information
                tools = parse_tool_info(msg["content"])
                
                # Create enhanced system message
                formatted_content = """[SYSTEM VERSION 1.3]
You are a home automation assistant. You must use the exact tool call format shown in the examples below.

Available tools and usage examples:

"""
                # Add tool definitions with examples
                for tool in tools:
                    formatted_content += format_tool_definition(
                        tool["name"],
                        tool["description"],
                        tool["parameters"]
                    )
                
                # Add conversation examples
                formatted_content += add_conversation_examples()
                
                system_message = {
                    "role": "system",
                    "content": formatted_content.strip()
                }
            else:
                # This is a tool response system message
                other_messages.append({
                    "role": "system",
                    "content": msg["content"]
                })
        else:
            if msg["role"] == "user":
                other_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                if "function_call" in msg:
                    # Convert function call to improved tool call format
                    tool_name = msg["function_call"]["name"]
                    args = json.loads(msg["function_call"]["arguments"])
                    tool_call = format_tool_call(tool_name, args)
                    
                    # Add natural language response before tool call
                    content = "I'll help you with that.\n" + tool_call
                    other_messages.append({
                        "role": "assistant",
                        "content": content
                    })
                else:
                    other_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
            elif msg["role"] == "function":
                # Convert function response to system message with compact format
                response_data = json.loads(msg["content"])
                formatted_response = f"<tool_response name='{msg['name']}'>{json.dumps(response_data, separators=(',', ':'))}</tool_response>"
                other_messages.append({
                    "role": "system",
                    "content": formatted_response
                })
    
    # Add system message first if it exists
    if system_message:
        mlx_messages.append(system_message)
    
    # Add all other messages in order
    mlx_messages.extend(other_messages)
    
    return {"messages": mlx_messages}

def process_jsonl_file(input_path: Path, output_path: Path) -> None:
    """Process a single JSONL file to enhanced MLX format."""
    logger.info(f"Processing {input_path} -> {output_path}")
    
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    
    # Convert each example
    with open(output_path, 'w') as f:
        for example in examples:
            enhanced_example = convert_to_mlx_format_v2(example)
            f.write(json.dumps(enhanced_example) + '\n')
    
    logger.info(f"Processed {len(examples)} examples")

def main():
    parser = argparse.ArgumentParser(description='Convert MLX format to enhanced MLX format')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Input directory containing MLX format JSONL files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for enhanced MLX format')
    
    args = parser.parse_args()
    
    # Use relative paths from current working directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process train and validation files
        for filename in ["train.jsonl", "valid.jsonl"]:
            input_file = input_dir / filename
            if input_file.exists():
                output_file = output_dir / filename
                process_jsonl_file(input_file, output_file)
            else:
                logger.warning(f"Input file not found: {input_file}")
            
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 