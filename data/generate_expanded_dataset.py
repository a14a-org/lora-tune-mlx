#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tool definitions with OpenAI-style function schema
TOOLS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather for"
                }
            },
            "required": ["location"]
        }
    },
    "control_lights": {
        "name": "control_lights",
        "description": "Control smart lights in a room",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "description": "The room name"
                },
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["on", "off", "dim"]
                },
                "brightness": {
                    "type": "integer",
                    "description": "Brightness level (0-100), required for dim action",
                    "minimum": 0,
                    "maximum": 100
                }
            },
            "required": ["room", "action"]
        }
    },
    "set_thermostat": {
        "name": "set_thermostat",
        "description": "Set thermostat temperature",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "description": "Target temperature in Celsius"
                }
            },
            "required": ["temperature"]
        }
    },
    "set_thermostat_delta": {
        "name": "set_thermostat_delta",
        "description": "Adjust temperature relative to current setting",
        "parameters": {
            "type": "object",
            "properties": {
                "delta": {
                    "type": "number",
                    "description": "Temperature change (+/- degrees)"
                }
            },
            "required": ["delta"]
        }
    }
}

# Error definitions for each tool
ERROR_TYPES = {
    "get_weather": {
        "location_not_found": {
            "status": "error",
            "error": {
                "code": "LOCATION_NOT_FOUND",
                "message": "Could not find weather data for location: {location}"
            }
        },
        "service_unavailable": {
            "status": "error",
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "message": "Weather service is temporarily unavailable"
            }
        }
    },
    "control_lights": {
        "room_not_found": {
            "status": "error",
            "error": {
                "code": "ROOM_NOT_FOUND",
                "message": "Room '{room}' not found in the system"
            }
        },
        "device_offline": {
            "status": "error",
            "error": {
                "code": "DEVICE_OFFLINE",
                "message": "Light control in {room} is currently offline"
            }
        }
    },
    "set_thermostat": {
        "invalid_temperature": {
            "status": "error",
            "error": {
                "code": "INVALID_TEMPERATURE",
                "message": "Temperature {temperature}°C is outside valid range (10-30°C)"
            }
        },
        "device_offline": {
            "status": "error",
            "error": {
                "code": "DEVICE_OFFLINE",
                "message": "Thermostat is currently offline"
            }
        }
    },
    "set_thermostat_delta": {
        "device_offline": {
            "status": "error",
            "error": {
                "code": "DEVICE_OFFLINE",
                "message": "Thermostat is currently offline"
            }
        }
    }
}

# Enhanced data for more natural variations
LOCATIONS = {
    "popular": ["London", "New York", "Paris", "Tokyo", "Sydney", "Berlin", "Amsterdam"],
    "us_cities": ["Chicago", "Los Angeles", "Miami", "Seattle", "Boston", "San Francisco"],
    "european": ["Rome", "Madrid", "Vienna", "Copenhagen", "Stockholm", "Dublin"],
    "asian": ["Singapore", "Hong Kong", "Seoul", "Bangkok", "Mumbai", "Dubai"],
    "error_prone": ["Unknown City", "Invalid Location", "NonexistentTown", "404City"]
}

ROOMS = {
    "common": ["living room", "bedroom", "kitchen", "bathroom", "office"],
    "specific": ["master bedroom", "guest room", "dining room", "study", "garage"],
    "error_prone": ["attic", "basement", "garden shed", "pool house", "invalid room"]
}

WEATHER_CONDITIONS = {
    "basic": ["sunny", "cloudy", "rainy", "partly cloudy", "clear"],
    "detailed": ["mostly sunny", "light rain", "heavy clouds", "scattered showers", "overcast"],
    "extreme": ["stormy", "foggy", "snowing", "thunderstorms", "windy"]
}

TIME_CONTEXTS = [
    "morning", "afternoon", "evening", "night",
    "dawn", "dusk", "midnight", "noon"
]

SEASONS = ["spring", "summer", "autumn", "winter"]

# Natural language variations for queries
QUERY_VARIATIONS = {
    "get_weather": {
        "basic": [
            "What's the weather like in {location}?",
            "How's the weather in {location}?",
            "Tell me the weather in {location}",
            "What's the temperature in {location}?"
        ],
        "contextual": [
            "Is it cold in {location} right now?",
            "Should I bring an umbrella in {location}?",
            "What's it like outside in {location}?",
            "How hot is it in {location} today?"
        ],
        "time_aware": [
            "What's the weather like in {location} this {time_context}?",
            "How's {location} looking this {time_context}?",
            "Tell me about the weather in {location} this {season}"
        ]
    },
    "control_lights": {
        "basic": [
            "Turn {action} the lights in the {room}",
            "Can you {action} the {room} lights?",
            "{action} the lights in the {room}"
        ],
        "contextual": [
            "It's getting dark in the {room}, can you turn on the lights?",
            "The {room} is too bright, dim the lights please",
            "I'm leaving the {room}, turn off the lights"
        ],
        "brightness": [
            "Set the {room} lights to {brightness}%",
            "Dim the {room} lights to {brightness}%",
            "Make the {room} lights {brightness}% bright"
        ]
    },
    "set_thermostat": {
        "basic": [
            "Set the temperature to {temperature}°C",
            "Change thermostat to {temperature} degrees",
            "Make it {temperature} degrees"
        ],
        "contextual": [
            "It's too {temp_feeling}, set it to {temperature}°C",
            "Can you make it {temperature} degrees? It's {temp_feeling} in here",
            "The {room} is {temp_feeling}, set temperature to {temperature} degrees"
        ]
    },
    "set_thermostat_delta": {
        "increase": [
            "Make it a bit warmer",
            "Can you increase the temperature slightly?",
            "It's a bit chilly, warm it up",
            "Turn up the heat a little"
        ],
        "decrease": [
            "Cool it down a bit",
            "Can you lower the temperature slightly?",
            "It's too warm, cool it down",
            "Make it a bit cooler"
        ]
    }
}

# Response templates for different scenarios
RESPONSE_TEMPLATES = {
    "get_weather": {
        "success": [
            "The temperature in {location} is {temperature}°C and it's {conditions}.",
            "Right now in {location}, it's {temperature}°C with {conditions} conditions.",
            "Current weather in {location}: {temperature}°C, {conditions}.",
            "In {location}, you can expect {conditions} weather with temperatures around {temperature}°C."
        ],
        "error": {
            "location_not_found": [
                "I couldn't find weather data for {location}. Are you sure that's the correct city name?",
                "Sorry, I wasn't able to get weather information for {location}. Could you verify the location?",
                "I couldn't locate weather data for {location}. Please check the spelling or try another nearby city."
            ],
            "service_unavailable": [
                "I'm having trouble connecting to the weather service right now. Please try again in a moment.",
                "The weather service is temporarily unavailable. Could you try again shortly?",
                "I can't access weather data at the moment due to a service interruption."
            ]
        }
    },
    "control_lights": {
        "success": {
            "on": [
                "I've turned on the lights in the {room}.",
                "The {room} lights are now on.",
                "Lights in the {room} have been switched on."
            ],
            "off": [
                "I've turned off the lights in the {room}.",
                "The {room} lights are now off.",
                "Lights in the {room} have been switched off."
            ],
            "dim": [
                "I've set the {room} lights to {brightness}% brightness.",
                "The {room} lights are now dimmed to {brightness}%.",
                "Brightness in the {room} is now at {brightness}%."
            ]
        },
        "error": {
            "room_not_found": [
                "I couldn't find a room called '{room}' in the system. Could you check the room name?",
                "The room '{room}' isn't set up in the system. Please verify the room name.",
                "I don't see '{room}' in the list of configured rooms."
            ],
            "device_offline": [
                "The lights in the {room} are currently offline. Please check the connection.",
                "I can't control the {room} lights right now as they appear to be disconnected.",
                "There seems to be a connection issue with the {room} lights."
            ]
        }
    },
    "set_thermostat": {
        "success": [
            "I've set the temperature to {temperature}°C.",
            "The thermostat is now set to {temperature}°C.",
            "Temperature has been changed to {temperature}°C.",
            "Done! The temperature is now set to {temperature}°C."
        ],
        "error": {
            "invalid_temperature": [
                "Sorry, {temperature}°C is outside the valid range (10-30°C). Please choose a temperature between these values.",
                "I can't set the temperature to {temperature}°C as it's outside the allowed range of 10-30°C.",
                "The temperature must be between 10°C and 30°C. {temperature}°C is not within this range."
            ],
            "device_offline": [
                "The thermostat is currently offline. Please check its connection.",
                "I can't adjust the temperature as the thermostat appears to be disconnected.",
                "There seems to be a connection issue with the thermostat."
            ]
        }
    },
    "set_thermostat_delta": {
        "success": {
            "increase": [
                "I've increased the temperature by {delta}°C to {temperature}°C.",
                "The temperature has been raised by {delta}°C. It's now {temperature}°C.",
                "Done! I've made it {delta}°C warmer. Current temperature: {temperature}°C."
            ],
            "decrease": [
                "I've decreased the temperature by {delta}°C to {temperature}°C.",
                "The temperature has been lowered by {delta}°C. It's now {temperature}°C.",
                "Done! I've made it {delta}°C cooler. Current temperature: {temperature}°C."
            ]
        },
        "error": {
            "device_offline": [
                "The thermostat is currently offline. Please check its connection.",
                "I can't adjust the temperature as the thermostat appears to be disconnected.",
                "There seems to be a connection issue with the thermostat."
            ]
        }
    }
}

def generate_system_message() -> str:
    """Generate the system message with tool definitions."""
    system_msg = "You are a home automation assistant. You have access to the following tools:\n\n"
    
    for tool in TOOLS.values():
        system_msg += f"{tool['name']}:\n"
        system_msg += f"  Description: {tool['description']}\n"
        system_msg += "  Parameters:\n"
        
        for param_name, param_info in tool['parameters']['properties'].items():
            required = "required" if param_name in tool['parameters'].get('required', []) else "optional"
            param_type = param_info['type']
            enum_values = f", values: {param_info['enum']}" if 'enum' in param_info else ""
            param_desc = param_info.get('description', '')
            system_msg += f"    * {param_name} ({param_type}, {required}{enum_values}): {param_desc}\n"
        
        system_msg += "\n"
    
    return system_msg

def generate_tool_response(tool_name: str, params: Dict[str, Any], should_error: bool = False) -> Dict[str, Any]:
    """Generate a response for a tool call, with potential errors."""
    if should_error:
        error_types = list(ERROR_TYPES[tool_name].keys())
        error_type = random.choice(error_types)
        error_template = ERROR_TYPES[tool_name][error_type]
        error_response = {
            "status": "error",
            "error": {
                "code": error_template["error"]["code"],
                "message": error_template["error"]["message"].format(**params)
            }
        }
        return error_response

    if tool_name == "get_weather":
        return {
            "status": "success",
            "data": {
                "temperature": random.randint(-5, 35),
                "unit": "C",
                "conditions": random.choice(WEATHER_CONDITIONS["basic"] + WEATHER_CONDITIONS["detailed"])
            }
        }
    elif tool_name == "control_lights":
        return {
            "status": "success",
            "data": {
                "room": params["room"],
                "action": params["action"],
                "brightness": params.get("brightness", 100 if params["action"] == "on" else 0)
            }
        }
    elif tool_name == "set_thermostat":
        return {
            "status": "success",
            "data": {
                "temperature": params["temperature"]
            }
        }
    elif tool_name == "set_thermostat_delta":
        current_temp = random.randint(18, 25)
        new_temp = current_temp + params["delta"]
        return {
            "status": "success",
            "data": {
                "previous_temperature": current_temp,
                "temperature": new_temp,
                "delta": params["delta"]
            }
        }

def format_function_call(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Format a function call in Qwen's expected format."""
    return {
        "name": tool_name,
        "arguments": json.dumps(params, ensure_ascii=False)
    }

def get_natural_response(tool_name: str, params: Dict[str, Any], response_data: Dict[str, Any]) -> str:
    """Get a natural language response based on the tool response."""
    if response_data["status"] == "error":
        error_code = response_data["error"]["code"]
        error_type = error_code.lower()
        templates = RESPONSE_TEMPLATES[tool_name]["error"][error_type]
        return random.choice(templates).format(**params)

    templates = RESPONSE_TEMPLATES[tool_name]["success"]
    if tool_name == "control_lights":
        templates = templates[params["action"]]
    elif tool_name == "set_thermostat_delta":
        templates = templates["increase" if params["delta"] > 0 else "decrease"]

    return random.choice(templates).format(**{**params, **response_data["data"]})

def generate_single_turn_example(tool_name: str, should_error: bool = False) -> Dict[str, Any]:
    """Generate a single-turn conversation example."""
    # Choose appropriate parameters based on tool
    if tool_name == "get_weather":
        location_list = (LOCATIONS["error_prone"] if should_error else 
                        random.choice([LOCATIONS["popular"], LOCATIONS["us_cities"], 
                                     LOCATIONS["european"], LOCATIONS["asian"]]))
        params = {"location": random.choice(location_list)}
        variations = QUERY_VARIATIONS["get_weather"]
        
        # Add weather-specific context
        context_params = {
            "time_context": random.choice(TIME_CONTEXTS),
            "season": random.choice(SEASONS)
        }
        
    elif tool_name == "control_lights":
        room_list = ROOMS["error_prone"] if should_error else ROOMS["common"] + ROOMS["specific"]
        action = random.choice(["on", "off", "dim"])
        params = {
            "room": random.choice(room_list),
            "action": action
        }
        if action == "dim":
            params["brightness"] = random.randint(1, 100)
        variations = QUERY_VARIATIONS["control_lights"]
        
        # Add light-specific context
        context_params = {
            "time_context": random.choice(TIME_CONTEXTS)
        }
        
    elif tool_name == "set_thermostat":
        temperature = random.randint(5, 35) if should_error else random.randint(18, 25)
        params = {"temperature": temperature}
        variations = QUERY_VARIATIONS["set_thermostat"]
        
        # Add temperature-specific context
        context_params = {
            "temp_feeling": random.choice(["hot", "cold", "warm", "cool"]),
            "room": random.choice(ROOMS["common"])  # Optional room context
        }
        
    else:  # set_thermostat_delta
        delta = random.choice([-2, -1.5, -1, 1, 1.5, 2])
        params = {"delta": delta}
        variations = QUERY_VARIATIONS["set_thermostat_delta"]["increase" if delta > 0 else "decrease"]
        
        # Add temperature-specific context
        context_params = {
            "temp_feeling": random.choice(["hot", "cold", "warm", "cool"])
        }

    # Select query template and format it
    if tool_name == "set_thermostat_delta":
        # For thermostat delta, we already selected the appropriate variation list
        query_template = random.choice(variations)
    else:
        # For other tools, select between basic and contextual
        if random.random() < 0.7:  # 70% basic queries, 30% contextual
            template_type = "basic"
        else:
            template_type = random.choice(["contextual", "time_aware"]) if "time_aware" in variations else "contextual"
        query_template = random.choice(variations[template_type])
    
    # Only use context parameters that are actually in the template
    template_params = {k: v for k, v in {**params, **context_params}.items() 
                      if "{" + k + "}" in query_template}
    query = query_template.format(**template_params)

    # Generate tool response and natural language response
    tool_response = generate_tool_response(tool_name, params, should_error)
    natural_response = get_natural_response(tool_name, params, tool_response)

    return {
        "messages": [
            {
                "role": "system",
                "content": generate_system_message()
            },
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": "",
                "function_call": format_function_call(tool_name, params)
            },
            {
                "role": "function",
                "name": tool_name,
                "content": json.dumps(tool_response, ensure_ascii=False)
            },
            {
                "role": "assistant",
                "content": natural_response
            }
        ]
    }

def generate_dataset(num_examples: int, output_dir: Path, split_ratio: float = 0.9) -> None:
    """Generate a dataset with the specified number of examples and distribution."""
    examples = []
    distribution = {
        "get_weather": {"total": int(0.3 * num_examples), "error_rate": 0.2},
        "control_lights": {"total": int(0.3 * num_examples), "error_rate": 0.2},
        "set_thermostat": {"total": int(0.2 * num_examples), "error_rate": 0.2},
        "set_thermostat_delta": {"total": int(0.2 * num_examples), "error_rate": 0.2}
    }
    
    logger.info(f"Generating {num_examples} examples...")
    logger.info(f"Output directory: {output_dir}")
    
    # Generate examples according to distribution
    for tool_name, config in distribution.items():
        total = config["total"]
        error_count = int(total * config["error_rate"])
        success_count = total - error_count
        
        logger.info(f"Generating {total} examples for {tool_name} ({error_count} errors)")
        
        # Generate success cases
        for _ in range(success_count):
            examples.append(generate_single_turn_example(tool_name, should_error=False))
        
        # Generate error cases
        for _ in range(error_count):
            examples.append(generate_single_turn_example(tool_name, should_error=True))
    
    # Shuffle examples
    random.shuffle(examples)
    
    # Split into train and validation sets
    split_idx = int(len(examples) * split_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Save datasets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / f"train_{timestamp}.json"
    valid_file = output_dir / f"validation_{timestamp}.json"
    
    logger.info(f"Saving training data to {train_file}")
    with open(train_file, 'w') as f:
        json.dump({"data": train_examples}, f, indent=2)
    
    logger.info(f"Saving validation data to {valid_file}")
    with open(valid_file, 'w') as f:
        json.dump({"data": val_examples}, f, indent=2)
    
    logger.info(f"Generated {len(train_examples)} training and {len(val_examples)} validation examples")
    logger.info(f"Files saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate expanded training data for Qwen-style function calling')
    parser.add_argument('--num-examples', type=int, default=1000,
                      help='Number of examples to generate')
    parser.add_argument('--output-dir', type=str, default='generated_qwen_expanded',
                      help='Output directory for generated data')
    parser.add_argument('--split-ratio', type=float, default=0.9,
                      help='Train/validation split ratio')
    
    args = parser.parse_args()
    
    # Use relative path from current working directory
    output_dir = Path(args.output_dir)
    
    logger.info(f"Output directory: {output_dir}")
    
    try:
        generate_dataset(args.num_examples, output_dir, args.split_ratio)
    except Exception as e:
        logger.error(f"Error generating dataset: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 