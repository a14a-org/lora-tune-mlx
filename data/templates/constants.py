"""Constants and templates for conversation generation."""

# Tool definitions with OpenAI-style function schema
TOOLS = {
    "list_dir": {
        "name": "list_dir",
        "description": "List directory contents",
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
        "description": "Read file contents",
        "parameters": {
            "type": "object",
            "properties": {
                "relative_workspace_path": {
                    "type": "string",
                    "description": "Path to read"
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
            "required": ["relative_workspace_path", "should_read_entire_file", "start_line_one_indexed", "end_line_one_indexed_inclusive"]
        }
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Edit file contents",
        "parameters": {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "File to edit"
                },
                "instructions": {
                    "type": "string",
                    "description": "Edit instructions"
                },
                "code_edit": {
                    "type": "string",
                    "description": "Code changes"
                }
            },
            "required": ["target_file", "instructions", "code_edit"]
        }
    },
    "run_terminal_cmd": {
        "name": "run_terminal_cmd",
        "description": "Run terminal command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run"
                },
                "is_background": {
                    "type": "boolean",
                    "description": "Run in background"
                },
                "require_user_approval": {
                    "type": "boolean",
                    "description": "Require approval"
                }
            },
            "required": ["command", "is_background", "require_user_approval"]
        }
    }
}

# Common file and directory patterns
COMMON_FILES = {
    "python": ["setup.py", "requirements.txt", "README.md", "__init__.py", "main.py", "utils.py"],
    "tests": ["test_*.py", "conftest.py"],
    "docs": ["docs/*.md", "*.rst"],
    "config": ["config/*.json", "*.yaml", "*.ini"]
}

COMMON_DIRS = {
    "python": ["src", "tests", "docs", "examples"],
    "web": ["frontend", "backend", "static", "templates"],
    "data": ["data", "datasets", "models", "checkpoints"]
}

# Python-specific constants
PYTHON_FILES = [
    "main.py", "utils.py", "helpers.py", "core.py", "app.py",
    "config.py", "database.py", "models.py", "views.py", "forms.py",
    "auth.py", "api.py", "tasks.py", "cache.py", "logger.py",
    "validation.py", "serializers.py", "handlers.py", "middleware.py", "settings.py"
]

COMMON_FUNCTIONS = [
    "calculate_average", "process_data", "validate_input", "format_output", "parse_config",
    "fetch_user_data", "update_database", "cache_results", "authenticate_request", "handle_errors",
    "transform_json", "validate_schema", "process_batch", "generate_report", "analyze_metrics",
    "filter_results", "sort_data", "merge_configs", "load_settings", "save_state",
    "compute_statistics", "normalize_data", "extract_features", "aggregate_results", "validate_permissions",
    "format_response", "parse_arguments", "initialize_system", "cleanup_resources", "handle_timeout"
]

# Response templates
RESPONSE_TEMPLATES = {
    "list_dir": {
        "success": [
            "Here's what I found in the {path} directory:\n{contents}",
            "The {path} directory contains:\n{contents}",
            "Contents of {path}:\n{contents}"
        ],
        "empty": [
            "The {path} directory is empty.",
            "No files found in {path}.",
            "The directory {path} contains no files or subdirectories."
        ],
        "error": [
            "Sorry, I couldn't access the {path} directory.",
            "There was an error reading the {path} directory.",
            "Failed to list contents of {path}."
        ]
    },
    "read_file": {
        "success": [
            "Here's the content of {file}:\n{content}",
            "I found the following in {file}:\n{content}",
            "Contents of {file}:\n{content}"
        ],
        "error": [
            "Sorry, I couldn't read {file}.",
            "There was an error reading {file}.",
            "Failed to access {file}."
        ]
    },
    "edit_file": {
        "success": [
            "I've updated {file} with the changes.",
            "The changes have been applied to {file}.",
            "{file} has been modified successfully."
        ],
        "error": [
            "Sorry, I couldn't modify {file}.",
            "There was an error updating {file}.",
            "Failed to apply changes to {file}."
        ]
    }
}

# Function implementation templates
FUNCTION_TEMPLATES = {
    "data_processing": {
        "basic": {
            "template": """def {name}(data: List[float]) -> float:
    \"\"\"
    Calculate {operation} of the input data.
    
    Args:
        data: List of numerical values to process
        
    Returns:
        Processed result
        
    Raises:
        ValueError: If input data is empty or contains invalid values
    \"\"\"
    if not data:
        raise ValueError("Input data cannot be empty")
    return {operation_impl}""",
            "operations": {
                "average": "sum(data) / len(data)",
                "maximum": "max(data)",
                "minimum": "min(data)",
                "sum": "sum(data)"
            }
        },
        "intermediate": {
            "template": """def {name}(data: List[Union[int, float]], threshold: float = 0.0) -> Dict[str, float]:
    \"\"\"
    Process numerical data with statistical analysis.
    
    Args:
        data: List of numerical values to analyze
        threshold: Optional threshold for filtering values
        
    Returns:
        Dictionary containing statistical results
        
    Raises:
        ValueError: If input data is empty or invalid
        TypeError: If input types are incorrect
    \"\"\"
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    if not data:
        raise ValueError("Input data cannot be empty")
    if not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("All values must be numeric")
        
    filtered_data = [x for x in data if x > threshold]
    result = dict()
    result["mean"] = sum(filtered_data) / len(filtered_data) if filtered_data else 0
    result["max"] = max(filtered_data) if filtered_data else threshold
    result["count"] = len(filtered_data)
    return result""",
            "operations": {
                "analyze": "result",
                "compute": "result",
                "process": "result"
            }
        }
    },
    "file_operations": {
        "basic": {
            "template": """def {name}(filepath: str) -> {return_type}:
    \"\"\"
    {operation_desc}
    
    Args:
        filepath: Path to the target file
        
    Returns:
        {return_desc}
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If file access is denied
    \"\"\"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    {operation_impl}""",
            "operations": {
                "read_text": {
                    "return_type": "str",
                    "desc": "Read text content from a file",
                    "return_desc": "Content of the file as string",
                    "impl": """with open(filepath, 'r') as f:
        return f.read()"""
                },
                "count_lines": {
                    "return_type": "int",
                    "desc": "Count number of lines in a file",
                    "return_desc": "Number of lines in the file",
                    "impl": """with open(filepath, 'r') as f:
        return sum(1 for _ in f)"""
                }
            }
        }
    },
    "api_handling": {
        "basic": {
            "template": """async def {name}(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    \"\"\"
    {operation_desc}
    
    Args:
        url: API endpoint URL
        params: Optional query parameters
        
    Returns:
        API response data
        
    Raises:
        HTTPError: If the API request fails
        ValueError: If the URL is invalid
    \"\"\"
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL scheme")
    
    async with aiohttp.ClientSession() as session:
        async with session.{method}(url, params=params) as response:
            response.raise_for_status()
            return await response.json()"""
        }
    }
}

# Error handling patterns
ERROR_PATTERNS = {
    "type_check": {
        "single": "if not isinstance({var}, {type}):\n    raise TypeError(f\"{var} must be {type.__name__}\")",
        "multiple": "if not isinstance({var}, ({types})):\n    raise TypeError(f\"{var} must be one of: {type_names}\")"
    },
    "value_check": {
        "range": "if not {min_val} <= {var} <= {max_val}:\n    raise ValueError(f\"{var} must be between {min_val} and {max_val}\")",
        "positive": "if {var} <= 0:\n    raise ValueError(f\"{var} must be positive\")",
        "non_empty": "if not {var}:\n    raise ValueError(f\"{var} cannot be empty\")"
    },
    "file_check": {
        "exists": "if not os.path.exists({path}):\n    raise FileNotFoundError(f\"File not found: {path}\")",
        "is_file": "if not os.path.isfile({path}):\n    raise ValueError(f\"{path} is not a file\")"
    }
}

# Improvement types and their descriptions
IMPROVEMENT_TYPES = {
    "error_handling": {
        "description": "Add comprehensive error checking and exception handling",
        "patterns": ["type_check", "value_check", "file_check"]
    },
    "documentation": {
        "description": "Add detailed docstrings and type hints",
        "sections": ["Args", "Returns", "Raises", "Examples"]
    },
    "optimization": {
        "description": "Improve performance and efficiency",
        "techniques": ["caching", "algorithm_improvement", "data_structure_optimization"]
    },
    "testing": {
        "description": "Add unit tests and test cases",
        "types": ["unit_tests", "integration_tests", "edge_cases"]
    }
}

# Project structure templates
PROJECT_TEMPLATES = {
    "single_file": {
        "files": ["main.py"],
        "imports": ["from typing import *"]
    },
    "basic_package": {
        "files": [
            "setup.py",
            "README.md",
            "requirements.txt",
            "src/__init__.py",
            "src/main.py",
            "tests/__init__.py",
            "tests/test_main.py"
        ],
        "imports": [
            "from typing import *",
            "import pytest"
        ]
    }
}

# Common test patterns
TEST_TEMPLATES = {
    "unit_test": """def test_{function_name}_{scenario}():
    \"\"\"Test {function_name} with {scenario} case.\"\"\"
    {setup}
    {assertion}""",
    "exception_test": """def test_{function_name}_{error_type}():
    \"\"\"Test {function_name} raises {error_type}.\"\"\"
    with pytest.raises({error_type}):
        {test_code}"""
} 