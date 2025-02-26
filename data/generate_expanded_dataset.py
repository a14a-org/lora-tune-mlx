#!/usr/bin/env python3
import json
import random
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict

from generators.file_manipulation import FileManipulationGenerator
from generators.realistic_scenarios import RealisticScenarioGenerator
from data.utils.openai_variations import generate_message_variations, DEFAULT_MODEL
# Import other generators as they are created

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
    },
    "codebase_search": {
        "name": "codebase_search",
        "description": "Find snippets of code from the codebase most relevant to the search query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant code"
                },
                "target_directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns for directories to search over"
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used"
                }
            },
            "required": ["query"]
        }
    },
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "relative_workspace_path": {
                    "type": "string",
                    "description": "The path of the file to read"
                },
                "start_line_one_indexed": {
                    "type": "integer",
                    "description": "The one-indexed line number to start reading from"
                },
                "end_line_one_indexed_inclusive": {
                    "type": "integer",
                    "description": "The one-indexed line number to end reading at"
                },
                "should_read_entire_file": {
                    "type": "boolean",
                    "description": "Whether to read the entire file"
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used"
                }
            },
            "required": ["relative_workspace_path", "start_line_one_indexed", "end_line_one_indexed_inclusive", "should_read_entire_file"]
        }
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Edit or create a file",
        "parameters": {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "The path of the file to edit"
                },
                "instructions": {
                    "type": "string",
                    "description": "Instructions for the edit"
                },
                "code_edit": {
                    "type": "string",
                    "description": "The code changes to make"
                }
            },
            "required": ["target_file", "instructions", "code_edit"]
        }
    },
    "grep_search": {
        "name": "grep_search",
        "description": "Search for text patterns in files",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "include_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to include"
                },
                "exclude_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to exclude"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive"
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used"
                }
            },
            "required": ["query"]
        }
    },
    "file_search": {
        "name": "file_search",
        "description": "Search for files by name",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Fuzzy filename to search for"
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used"
                }
            },
            "required": ["query", "explanation"]
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

# Cost estimation constants (based on OpenAI pricing as of 2024)
MODEL_COSTS = {
    "gpt-3.5-turbo-instruct": 0.0015,  # Cost per 1K tokens
    "gpt-4-turbo-preview": 0.01,       # Cost per 1K input tokens
    "gpt-4": 0.03,                     # Cost per 1K input tokens
}

# Token estimation constants
TOKENS_PER_WORD = 1.3  # Average tokens per word
MIN_TOKENS_PER_REQUEST = 50  # Minimum tokens per API request for prompt

class CostTracker:
    """Track API usage and estimate costs."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.total_tokens = 0
        self.total_api_calls = 0
        self.variations_generated = 0
        
    def add_usage(self, tokens: int):
        """Add token usage."""
        # Add minimum prompt tokens plus estimated completion tokens
        total_tokens = MIN_TOKENS_PER_REQUEST + tokens
        self.total_tokens += total_tokens
        self.total_api_calls += 1
        self.variations_generated += 1
    
    def estimate_cost(self) -> float:
        """Estimate total cost in USD."""
        cost_per_1k = MODEL_COSTS.get(self.model, MODEL_COSTS[DEFAULT_MODEL])
        return (self.total_tokens / 1000) * cost_per_1k
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model": self.model,
            "total_tokens": self.total_tokens,
            "total_api_calls": self.total_api_calls,
            "variations_generated": self.variations_generated,
            "estimated_cost_usd": round(self.estimate_cost(), 4)
        }

def preview_variations(
    messages: List[Dict[str, str]],
    num_previews: int = 3,
    model: str = DEFAULT_MODEL
) -> Tuple[List[str], Dict[str, Any]]:
    """Preview message variations without generating full dataset.
    
    Args:
        messages: List of messages to generate variations for
        num_previews: Number of messages to preview
        model: OpenAI model to use
        
    Returns:
        Tuple of (variations list, cost estimation)
    """
    cost_tracker = CostTracker(model)
    variations = []
    
    # Only process a few messages for preview
    for i, msg in enumerate(messages[:num_previews]):
        if msg["role"] == "user":
            try:
                new_variations = generate_message_variations(
                    msg["content"],
                    num_variations=2,  # Generate fewer variations for preview
                    model=model
                )
                variations.extend(new_variations)
                # Estimate tokens based on input length and variations
                words = len(msg["content"].split())
                estimated_tokens = int(words * TOKENS_PER_WORD * 2)  # 2 variations
                cost_tracker.add_usage(estimated_tokens)
            except Exception as e:
                logger.warning(f"Error generating preview variation {i}: {e}")
    
    return variations, cost_tracker.get_stats()

def generate_dataset(
    num_examples: int,
    output_path: str,
    use_openai: bool = True,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    cost_tracker: Optional[CostTracker] = None
) -> Optional[Dict[str, Any]]:
    """Generate dataset with the specified number of examples.
    
    Args:
        num_examples: Number of examples to generate
        output_path: Path to save the generated dataset
        use_openai: Whether to use OpenAI API for generating message variations
        model: OpenAI model to use
        dry_run: If True, only preview variations without generating full dataset
        cost_tracker: Optional cost tracker to update
        
    Returns:
        If dry_run is True, returns dict with preview info and cost estimation
        Otherwise returns None after saving the dataset
    """
    logger.info(f"{'Previewing' if dry_run else 'Generating'} {num_examples} examples...")
    
    if dry_run:
        # Generate a few examples for preview
        scenario_generator = RealisticScenarioGenerator(use_openai_variations=use_openai, model=model)
        preview_examples = []
        for _ in range(min(3, num_examples)):
            example = scenario_generator.generate()
            preview_examples.append(example)
        
        # Preview variations for these examples
        all_variations = []
        for example in preview_examples:
            variations, stats = preview_variations(
                example["messages"],
                num_previews=2,
                model=model
            )
            all_variations.extend(variations)
        
        return {
            "num_examples_previewed": len(preview_examples),
            "variations_preview": all_variations[:5],  # Show first 5 variations
            "cost_estimation": {
                "per_example": stats["estimated_cost_usd"] / len(preview_examples),
                "total_estimated": stats["estimated_cost_usd"] * (num_examples / len(preview_examples))
            }
        }
    
    # Normal dataset generation
    if cost_tracker is None:
        cost_tracker = CostTracker(model)
        
    scenario_generator = RealisticScenarioGenerator(
        use_openai_variations=use_openai,
        model=model
    )
    examples = []
    
    for i in range(num_examples):
        example = scenario_generator.generate()
        examples.append(example)
        
        # Update progress every 10 examples
        if (i + 1) % 10 == 0:
            stats = cost_tracker.get_stats()
            logger.info(
                f"Generated {i + 1}/{num_examples} examples. "
                f"Estimated cost so far: ${stats['estimated_cost_usd']:.4f}"
            )
    
    # Save the dataset
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    stats = cost_tracker.get_stats()
    logger.info(
        f"Generated {len(examples)} examples and saved to {output_path}\n"
        f"Final statistics:\n"
        f"- Total tokens: {stats['total_tokens']:,}\n"
        f"- Total API calls: {stats['total_api_calls']:,}\n"
        f"- Total variations: {stats['variations_generated']:,}\n"
        f"- Total estimated cost: ${stats['estimated_cost_usd']:.4f}"
    )
    return None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate expanded dataset')
    parser.add_argument('--num_train', type=int, default=190,
                       help='Number of training examples')
    parser.add_argument('--num_valid', type=int, default=20,
                       help='Number of validation examples')
    parser.add_argument('--output_dir', type=str, default='data/multi_step',
                       help='Output directory')
    parser.add_argument('--use_openai', action='store_true',
                       help='Use OpenAI API for generating message variations')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help=f'OpenAI model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--dry_run', action='store_true',
                       help='Preview variations and estimate costs without generating full dataset')
    args = parser.parse_args()
    
    # Initialize cost tracker
    cost_tracker = CostTracker(args.model)
    
    if args.dry_run:
        # Preview training data generation
        logger.info("Previewing training data generation...")
        preview_info = generate_dataset(
            args.num_train,
            os.path.join(args.output_dir, 'train.json'),
            args.use_openai,
            model=args.model,
            dry_run=True
        )
        
        print("\nPreview Results:")
        print(f"Examples previewed: {preview_info['num_examples_previewed']}")
        print("\nSample variations:")
        for i, var in enumerate(preview_info['variations_preview'], 1):
            print(f"{i}. {var}")
        
        print("\nCost Estimation:")
        print(f"Cost per example: ${preview_info['cost_estimation']['per_example']:.4f}")
        total_examples = args.num_train + args.num_valid
        total_cost = preview_info['cost_estimation']['total_estimated']
        print(f"Total estimated cost for {total_examples} examples: ${total_cost:.4f}")
        
        return
    
    # Generate training data
    train_path = os.path.join(args.output_dir, 'train.json')
    generate_dataset(
        args.num_train,
        train_path,
        args.use_openai,
        model=args.model,
        cost_tracker=cost_tracker
    )
    print(f'Generated {args.num_train} training examples')
    
    # Generate validation data
    valid_path = os.path.join(args.output_dir, 'valid.json')
    generate_dataset(
        args.num_valid,
        valid_path,
        args.use_openai,
        model=args.model,
        cost_tracker=cost_tracker
    )
    print(f'Generated {args.num_valid} validation examples')
    
    # Print final statistics
    final_stats = cost_tracker.get_stats()
    print("\nFinal Generation Statistics:")
    print(f"Total tokens used: {final_stats['total_tokens']:,}")
    print(f"Total API calls: {final_stats['total_api_calls']:,}")
    print(f"Total variations generated: {final_stats['variations_generated']:,}")
    print(f"Total estimated cost: ${final_stats['estimated_cost_usd']:.4f}")

if __name__ == '__main__':
    main() 