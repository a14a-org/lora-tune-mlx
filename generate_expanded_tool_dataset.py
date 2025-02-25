#!/usr/bin/env python3
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from difflib import get_close_matches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tool definitions based on TOOL_USAGE.md
TOOLS = {
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

# Scenario type definitions
@dataclass
class ScenarioMetadata:
    """Metadata for a generated scenario"""
    scenario_type: str
    num_steps: int
    tools_used: List[str]
    success: bool
    error_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format"""
        return {
            "scenario_type": self.scenario_type,
            "num_steps": self.num_steps,
            "tools_used": self.tools_used,
            "success": self.success,
            "error_step": self.error_step
        }

class TemplateManager:
    """Manages templates for different types of content and queries"""
    
    def __init__(self):
        self.query_templates = {
            'check_existence': [
                "Can you check if {path} exists?",
                "Does {path} exist in the workspace?",
                "Look for {path} in the project",
                "Verify if {path} is present",
                "Check if we have {path} in our workspace",
                "Is there a file called {path}?",
                "Could you see if {path} exists?",
                "I need to know if {path} is in the project",
                "Search for a file named {path}",
                "Do we already have {path}?",
                "Can you find {path} in our codebase?",
                "Check the workspace for {path}",
                "See if you can locate {path}",
                "I'm looking for {path}, is it there?",
                "Does our project include {path}?"
            ],
            'create_file': [
                "Create a new file at {path}",
                "Make a file called {path}",
                "Set up a new file {path}",
                "Generate a file at {path}",
                "Create {path} with the following content",
                "I need a new file at {path}",
                "Could you create {path} for me?",
                "Please set up {path}",
                "Add a new file {path} to the project",
                "We need a file at {path}",
                "Initialize a new file called {path}",
                "Start a new file {path}",
                "Create {path} from scratch",
                "Set up {path} with some initial content",
                "Make a new {path} file"
            ],
            'read_content': [
                "What's in the file {path}?",
                "Show me the contents of {path}",
                "Read the file {path}",
                "Can you check what's inside {path}?",
                "Display the content of {path}",
                "What does {path} contain?",
                "Let me see what's in {path}",
                "Could you show me {path}'s contents?",
                "I need to see what's in {path}",
                "Print out the contents of {path}",
                "Show the code in {path}",
                "What's written in {path}?",
                "Read out {path} for me",
                "Get the contents of {path}",
                "Open {path} and show me what's inside"
            ],
            'update_content': [
                "Update {path} with the following changes",
                "Modify the content of {path}",
                "Make changes to {path}",
                "Edit {path} to include",
                "Replace the content in {path}",
                "I need to update {path}",
                "Could you modify {path} for me?",
                "Change the contents of {path}",
                "Add some new code to {path}",
                "Update the implementation in {path}",
                "Revise {path} with these changes",
                "Enhance the code in {path}",
                "Improve {path} by adding",
                "Refactor the code in {path}",
                "Make some improvements to {path}"
            ],
            'remove_file': [
                "Remove the file {path}",
                "Delete {path}",
                "Get rid of {path}",
                "Remove {path} from the workspace",
                "Delete the file at {path}",
                "Could you delete {path}?",
                "Please remove {path}",
                "I want to delete {path}",
                "Take out {path} from the project",
                "We don't need {path} anymore",
                "Clean up by removing {path}",
                "Drop {path} from the codebase",
                "Eliminate {path} from the project",
                "Can you erase {path}?",
                "Remove {path} as it's no longer needed"
            ],
            'project_setup': [
                "Create a new {project_type} project called {project_name}",
                "Set up a {project_type} project named {project_name}",
                "Initialize a new {project_type} project: {project_name}",
                "Start a {project_type} project with name {project_name}",
                "Create {project_name} as a {project_type} project",
                "Make a new {project_type} project {project_name}",
                "Begin a {project_type} project called {project_name}",
                "Set up {project_name} using {project_type}",
                "Generate a new {project_type} project: {project_name}",
                "Bootstrap a {project_type} project named {project_name}",
                "Create a {project_type} codebase called {project_name}",
                "Initialize {project_name} as a {project_type} project",
                "Start building {project_name} with {project_type}",
                "Set up a new {project_name} using {project_type}",
                "Create project {project_name} with {project_type}"
            ]
        }
        
        self.file_templates = {
            'python_script': {
                'basic': [
                    "#!/usr/bin/env python3\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()",
                    "#!/usr/bin/env python3\n\n# Main script functionality\n\ndef main():\n    # TODO: Implement main logic\n    pass\n\nif __name__ == '__main__':\n    main()"
                ],
                'with_argparse': [
                    """#!/usr/bin/env python3
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='{description}')
    parser.add_argument('--input', help='Input file path')
    return parser.parse_args()

def main():
    args = parse_args()
    # TODO: Implement main logic
    pass

if __name__ == '__main__':
    main()"""
                ]
            },
            'package_json': [
                '''{{
  "name": "{name}",
  "version": "1.0.0",
  "description": "{description}",
  "main": "index.js",
  "scripts": {{
    "test": "echo \\"Error: no test specified\\" && exit 1"
  }},
  "keywords": [],
  "author": "",
  "license": "ISC"
}}'''
            ],
            'readme_md': [
                """# {project_name}

{description}

## Installation

```bash
{install_command}
```

## Usage

```bash
{usage_command}
```

## License

{license}
"""
            ]
        }

class ErrorHandler:
    """Handles error scenarios and generates appropriate error responses"""
    
    def __init__(self):
        self.error_templates = {
            'path_not_found': [
                "The path {path} does not exist",
                "Could not find {path} in the workspace",
                "The specified path {path} was not found"
            ],
            'permission_denied': [
                "Permission denied when accessing {path}",
                "Cannot access {path} due to insufficient permissions",
                "Access to {path} is restricted"
            ],
            'invalid_content': [
                "The content for {path} is invalid",
                "Cannot process invalid content for {path}",
                "The provided content contains errors"
            ]
        }
    
    def generate_error_response(self, error_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an error response based on the error type"""
        template = random.choice(self.error_templates[error_type])
        return {
            "status": "error",
            "error": {
                "type": error_type,
                "message": template.format(**params)
            }
        }

class ScenarioGenerator:
    """Generates different types of multi-step scenarios"""
    
    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager
        
        # Common file paths and content types for scenarios
        self.common_paths = {
            'python_script': ['src/main.py', 'scripts/process.py', 'tools/utility.py', 'tests/test_main.py'],
            'package_json': ['package.json'],
            'readme_md': ['README.md', 'docs/README.md']
        }
        
        # Function call templates
        self.function_call_templates = {
            'list_dir': lambda path: {
                "name": "list_dir",
                "arguments": json.dumps({"relative_workspace_path": str(Path(path).parent)})
            },
            'read_file': lambda path: {
                "name": "read_file",
                "arguments": json.dumps({
                    "relative_workspace_path": path,
                    "should_read_entire_file": True,
                    "start_line_one_indexed": 1,
                    "end_line_one_indexed_inclusive": 1
                })
            },
            'edit_file': lambda path, content, instruction: {
                "name": "edit_file",
                "arguments": json.dumps({
                    "relative_workspace_path": path,
                    "target_file": path,
                    "instructions": instruction,
                    "code_edit": content
                })
            },
            'run_terminal_cmd': lambda cmd: {
                "name": "run_terminal_cmd",
                "arguments": json.dumps({
                    "command": cmd,
                    "is_background": False,
                    "require_user_approval": True
                })
            }
        }
        
        # Function response templates
        self.function_response_templates = {
            'list_dir': {
                'success': lambda path: {
                    "status": "success",
                    "data": {
                        "contents": []  # Empty directory
                    }
                },
                'error': lambda path: {
                    "status": "error",
                    "error": {
                        "type": "path_not_found",
                        "message": f"Directory {Path(path).parent} does not exist"
                    }
                },
                'similar_found': lambda path, similar_files: {
                    "status": "success",
                    "data": {
                        "contents": similar_files,
                        "message": f"Found similar files that might match your request"
                    }
                }
            },
            'read_file': {
                'success': lambda path: {
                    "status": "success",
                    "data": {
                        "content": ""  # File doesn't exist yet
                    }
                },
                'error': lambda path: {
                    "status": "error",
                    "error": {
                        "type": "path_not_found",
                        "message": f"File {path} does not exist"
                    }
                },
                'permission_error': lambda path: {
                    "status": "error",
                    "error": {
                        "type": "permission_denied",
                        "message": f"Permission denied when trying to read {path}"
                    }
                }
            },
            'edit_file': {
                'success': lambda path: {
                    "status": "success",
                    "data": {
                        "message": f"Successfully created file {path}"
                    }
                },
                'error': lambda path: {
                    "status": "error",
                    "error": {
                        "type": "write_error",
                        "message": f"Failed to write to file {path}"
                    }
                },
                'permission_error': lambda path: {
                    "status": "error",
                    "error": {
                        "type": "permission_denied",
                        "message": f"Permission denied when trying to write to {path}"
                    }
                },
                'syntax_error': lambda path: {
                    "status": "error",
                    "error": {
                        "type": "syntax_error",
                        "message": f"Syntax error in the content for {path}"
                    }
                }
            },
            'run_terminal_cmd': {
                'success': lambda cmd: {
                    "status": "success",
                    "data": {
                        "stdout": "" if cmd.startswith("mkdir") else (
                            "Initialized empty Git repository" if "git init" in cmd else ""
                        ),
                        "stderr": ""
                    }
                },
                'error': lambda cmd: {
                    "status": "error",
                    "error": {
                        "type": "command_error",
                        "message": (
                            f"mkdir: {cmd.split()[-1]}: Permission denied" if cmd.startswith("mkdir") else
                            f"git: '{cmd}': Permission denied" if "git" in cmd else
                            f"Command failed: {cmd}"
                        )
                    }
                },
                'not_found': lambda cmd: {
                    "status": "error",
                    "error": {
                        "type": "command_not_found",
                        "message": f"Command not found: {cmd.split()[0]}"
                    }
                },
                'timeout': lambda cmd: {
                    "status": "error",
                    "error": {
                        "type": "timeout",
                        "message": f"Command timed out after 30 seconds: {cmd}"
                    }
                }
            }
        }
    
    def _generate_assistant_response(self, scenario_type: str, step: int, success: bool, params: Dict[str, Any]) -> str:
        """Generate appropriate assistant response based on scenario type and step"""
        if not success:
            if 'error_type' in params:
                if params['error_type'] == 'path_not_found':
                    if 'similar_file' in params:
                        return f"I couldn't find {params['path']}, but I found a similar file: {params['similar_file']}. Would you like me to work with that file instead?"
                    return f"I couldn't find {params['path']}. Would you like me to create it?"
                elif params['error_type'] == 'permission_denied':
                    return f"I don't have permission to access {params['path']}. You might need to adjust the file permissions."
                elif params['error_type'] == 'syntax_error':
                    return f"There was a syntax error in the content for {params['path']}. I'll try to fix it and try again."
                elif params['error_type'] == 'command_not_found':
                    return f"The command {params.get('command', '').split()[0]} is not available. We might need to install it first."
                elif params['error_type'] == 'timeout':
                    return f"The command timed out. Would you like me to try again or use a different approach?"
            
            return f"I encountered an error: {params.get('error_message', 'An unknown error occurred')}. Would you like me to try a different approach?"
            
        if scenario_type == 'file_creation':
            if step == 1:  # After checking existence
                responses = [
                    f"I see that {params['path']} doesn't exist yet. I'll create it for you.",
                    f"The file {params['path']} hasn't been created yet. Let me set that up.",
                    f"I couldn't find {params['path']}. I'll create it now.",
                    f"{params['path']} is missing. I'll generate it for you.",
                    f"There's no {params['path']} yet. I'll make one."
                ]
            elif step == 2:  # After creating directory
                responses = [
                    f"I've created the necessary directory. Now I'll create the file with the content you specified.",
                    f"Directory structure is ready. Let's create the file with your content.",
                    f"Folder created successfully. Moving on to creating the file.",
                    f"Directory path is set up. Now for the file content.",
                    f"Path prepared. Time to add the file content."
                ]
            elif step == 3:  # After creating file
                responses = [
                    f"I've successfully created {params['path']} with the specified content.",
                    f"Done! {params['path']} has been created with your content.",
                    f"The file {params['path']} is now ready with the content you wanted.",
                    f"Created {params['path']} successfully with the specified content.",
                    f"All done! You can now find your content in {params['path']}."
                ]
            return random.choice(responses)
            
        elif scenario_type == 'file_update':
            if step == 1:  # After checking existence
                if 'similar_file' in params:
                    return f"I found a similar file {params['similar_file']} instead of {params['path']}. I'll proceed with that one."
                responses = [
                    f"I found {params['path']}. I'll read its content and make the requested changes.",
                    f"Located {params['path']}. Let me check its contents and update it.",
                    f"The file {params['path']} exists. I'll proceed with the modifications.",
                    f"Found {params['path']}. I'll read it and apply your changes.",
                    f"{params['path']} is here. Let me update it for you."
                ]
            elif step == 2:  # After reading content
                responses = [
                    f"I've read the current content. I'll now update it with the changes you specified.",
                    f"Got the current content. Making your requested changes now.",
                    f"Content loaded. Applying your modifications.",
                    f"I see the current version. Let me add your changes.",
                    f"Current content retrieved. Time to update it."
                ]
            elif step == 3:  # After updating content
                responses = [
                    f"I've successfully updated {params['path']} with the new content.",
                    f"Changes have been applied to {params['path']} successfully.",
                    f"Done! {params['path']} has been updated with your changes.",
                    f"The file {params['path']} now includes your modifications.",
                    f"Update complete! {params['path']} has been modified as requested."
                ]
            return random.choice(responses)
            
        elif scenario_type == 'project_setup':
            if step == 1:  # After checking project directory
                responses = [
                    f"I'll set up a new project called {params['project_name']}. First, I'll create the project structure.",
                    f"Let's create the {params['project_name']} project. Starting with the basic structure.",
                    f"Creating a new project {params['project_name']}. First step: project scaffolding.",
                    f"I'll help you set up {params['project_name']}. Let's start with the directory structure.",
                    f"Setting up project {params['project_name']}. Beginning with the foundation."
                ]
            elif step == 2:  # After creating project directory
                responses = [
                    "Project directory created. Now I'll initialize git and create the basic project files.",
                    "Directory is ready. Let's set up version control and project files.",
                    "Project folder created. Moving on to git initialization and file setup.",
                    "Basic structure ready. Time for git setup and initial files.",
                    "Project space prepared. Next: version control and project files."
                ]
            elif step == 3:  # After git init
                responses = [
                    "Git repository initialized. I'll now create the package configuration file.",
                    "Version control is set up. Moving on to package configuration.",
                    "Git is ready. Let's set up the project configuration.",
                    "Repository initialized. Time for package setup.",
                    "Git setup complete. Creating project configuration now."
                ]
            elif step == 4:  # After creating package file
                responses = [
                    "Package configuration created. Setting up the project documentation.",
                    "Config file is ready. Let's add some documentation.",
                    "Package setup done. Time to create the documentation.",
                    "Configuration complete. Moving on to documentation.",
                    "Project config is set. Adding documentation files."
                ]
            elif step == 5:  # After creating README
                responses = [
                    "Documentation added. Creating the initial source files.",
                    "Docs are ready. Let's set up the source code structure.",
                    "README is in place. Time for the source files.",
                    "Documentation complete. Adding source code files.",
                    "Docs created. Setting up the code structure."
                ]
            elif step == 6:  # After creating source files
                responses = [
                    f"Project {params['project_name']} has been successfully set up with all necessary files and structure.",
                    f"All done! {params['project_name']} is ready for development.",
                    f"Setup complete! {params['project_name']} has all the essential components.",
                    f"Finished setting up {params['project_name']} with everything you need.",
                    f"{params['project_name']} is now ready to go with all required files and structure."
                ]
            return random.choice(responses)
                
        return random.choice([
            "Operation completed successfully.",
            "Task finished successfully.",
            "All done! Everything worked as expected.",
            "Completed the operation successfully.",
            "Task completed without any issues."
        ])

    def _generate_file_modifications(self, content_type: str, original_content: str) -> Tuple[str, str]:
        """Generate modifications for a file based on its type and content"""
        if content_type == 'python_script':
            # Add a new function or modify existing one
            new_function = """
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Process input data and return results\"\"\"
    result = {}
    for key, value in data.items():
        result[key] = str(value).upper()
    return result
"""
            instruction = "Add new data processing function"
            modified_content = original_content + "\n" + new_function
            
        elif content_type == 'package_json':
            # Add new dependency
            package_data = json.loads(original_content)
            if 'dependencies' not in package_data:
                package_data['dependencies'] = {}
            package_data['dependencies']['lodash'] = '^4.17.21'
            modified_content = json.dumps(package_data, indent=2)
            instruction = "Add lodash dependency"
            
        else:  # readme_md
            # Add new section
            new_section = """
## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
"""
            instruction = "Add contributing section"
            modified_content = original_content + new_section
            
        return instruction, modified_content

    def _get_project_structure(self, project_type: str) -> Dict[str, Any]:
        """Generate project structure based on project type"""
        if project_type == 'python':
            return {
                'directories': [
                    'src',
                    'tests',
                    'docs',
                    'scripts'
                ],
                'files': {
                    'requirements.txt': """
# Core dependencies
pytest>=7.0.0
black>=22.0.0
mypy>=1.0.0
""".strip(),
                    'README.md': self.template_manager.file_templates['readme_md'][0],
                    'src/__init__.py': "",
                    'src/main.py': self.template_manager.file_templates['python_script']['basic'][1],
                    'tests/__init__.py': "",
                    'tests/test_main.py': """#!/usr/bin/env python3
import pytest
from src.main import main

def test_main():
    # TODO: Add test cases
    pass
""".strip(),
                    '.gitignore': """
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
""".strip()
                }
            }
        else:  # nodejs
            return {
                'directories': [
                    'src',
                    'test',
                    'docs'
                ],
                'files': {
                    'package.json': self.template_manager.file_templates['package_json'][0],
                    'README.md': self.template_manager.file_templates['readme_md'][0],
                    'src/index.js': """
const main = () => {
    console.log('Hello from the project!');
};

if (require.main === module) {
    main();
}

module.exports = { main };
""".strip(),
                    'test/index.test.js': """
const { main } = require('../src/index');

describe('Main module', () => {
    test('should run without errors', () => {
        expect(() => main()).not.toThrow();
    });
});
""".strip(),
                    '.gitignore': """
node_modules/
coverage/
dist/
build/
.env
*.log
""".strip()
                }
            }

    def _handle_file_not_found(self, target_path: str, existing_files: List[str]) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
        """Handle file not found errors by suggesting similar files and generating recovery messages.
        
        Returns:
            Tuple containing:
            - bool: Whether a similar file was found
            - Optional[str]: The suggested file path if found
            - List[Dict[str, Any]]: Additional messages to add to the conversation
        """
        # Get the filename without directory
        target_filename = Path(target_path).name
        
        # Find similar filenames
        similar_files = get_close_matches(target_filename, existing_files, n=1, cutoff=0.6)
        
        if not similar_files:
            return False, None, [{
                "role": "assistant",
                "content": f"I couldn't find {target_path} or any similar files. Would you like me to create it?"
            }]
            
        suggested_file = similar_files[0]
        messages = [
            {
                "role": "assistant",
                "content": f"I couldn't find {target_path}, but I found a similar file: {suggested_file}. Would you like me to work with that file instead?"
            },
            {
                "role": "user",
                "content": "Yes, please use that file."
            },
            {
                "role": "assistant",
                "content": f"I'll proceed with {suggested_file} instead."
            }
        ]
        
        return True, suggested_file, messages

    def generate_file_creation_scenario(self, path: str = None, content_type: str = None) -> Dict[str, Any]:
        """Generate a file creation scenario with multiple steps"""
        # If path and content_type not provided, randomly select them
        if path is None or content_type is None:
            content_type = random.choice(list(self.common_paths.keys()))
            path = random.choice(self.common_paths[content_type])
            
        messages = []
        tools_used = []
        
        # Step 1: Check if file exists
        messages.append({
            "role": "user",
            "content": random.choice(self.template_manager.query_templates['check_existence']).format(path=path)
        })
        
        # Assistant checks directory existence
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": self.function_call_templates['list_dir'](path)
        })
        
        # Function response shows directory doesn't exist
        messages.append({
            "role": "function",
            "name": "list_dir",
            "content": json.dumps(self.function_response_templates['list_dir']['success'](path))
        })
        
        tools_used.append("list_dir")
        
        # Assistant response and next action
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('file_creation', 1, True, {'path': path})
        })
        
        # Step 2: Create directory if needed
        if '/' in path:
            dir_path = str(Path(path).parent)
            mkdir_cmd = f"mkdir -p {dir_path}"
            
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": self.function_call_templates['run_terminal_cmd'](mkdir_cmd)
            })
            
            messages.append({
                "role": "function",
                "name": "run_terminal_cmd",
                "content": json.dumps(self.function_response_templates['run_terminal_cmd']['success'](mkdir_cmd))
            })
            
            tools_used.append("run_terminal_cmd")
            
            messages.append({
                "role": "assistant",
                "content": self._generate_assistant_response('file_creation', 2, True, {'path': path})
            })
        
        # Step 3: Create file with content
        file_content = random.choice(self.template_manager.file_templates[content_type]['basic']
                                   if content_type == 'python_script'
                                   else self.template_manager.file_templates[content_type])
        
        # Format template if needed
        if content_type == 'python_script' and 'description' in file_content:
            file_content = file_content.format(description="A generated Python script")
        elif content_type == 'package_json':
            file_content = file_content.format(
                name=Path(path).stem,
                description="A generated package"
            )
        elif content_type == 'readme_md':
            file_content = file_content.format(
                project_name=Path(path).parent.name or "Project",
                description="A generated project",
                install_command="npm install" if content_type == 'package_json' else "pip install -r requirements.txt",
                usage_command="npm start" if content_type == 'package_json' else "python main.py",
                license="MIT"
            )
        
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": self.function_call_templates['edit_file'](
                path,
                file_content,
                f"Create new {content_type} file"
            )
        })
        
        messages.append({
            "role": "function",
            "name": "edit_file",
            "content": json.dumps(self.function_response_templates['edit_file']['success'](path))
        })
        
        tools_used.append("edit_file")
        
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('file_creation', 3, True, {'path': path})
        })
        
        return {
            "messages": messages,
            "metadata": ScenarioMetadata(
                scenario_type="file_creation",
                num_steps=len(tools_used),
                tools_used=tools_used,
                success=True
            ).to_dict()
        }
    
    def generate_file_update_scenario(self, path: str = None, content_type: str = None) -> Dict[str, Any]:
        """Generate a file update scenario with multiple steps"""
        # If path and content_type not provided, randomly select them
        if path is None or content_type is None:
            content_type = random.choice(list(self.common_paths.keys()))
            path = random.choice(self.common_paths[content_type])
            
        messages = []
        tools_used = []
        
        # Step 1: Check if file exists
        messages.append({
            "role": "user",
            "content": random.choice(self.template_manager.query_templates['update_content']).format(path=path)
        })
        
        # Randomly decide if we should simulate a typo in the file path
        should_simulate_typo = random.random() < 0.2  # 20% chance
        if should_simulate_typo:
            # Create a typo in the filename
            path_parts = path.rsplit('.', 1)
            if len(path_parts) > 1:
                typo_path = f"{path_parts[0][:-1]}x.{path_parts[1]}"  # Change last character before extension
            else:
                typo_path = f"{path[:-1]}x"  # Change last character
                
            # First check with typo
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": self.function_call_templates['list_dir'](typo_path)
            })
            
            # Function response shows directory exists but not the file
            messages.append({
                "role": "function",
                "name": "list_dir",
                "content": json.dumps(self.function_response_templates['list_dir']['similar_found'](
                    typo_path,
                    [Path(path).name]  # The correct filename
                ))
            })
            
            tools_used.append("list_dir")
            
            # Handle the error and get recovery messages
            _, suggested_file, recovery_messages = self._handle_file_not_found(typo_path, [Path(path).name])
            messages.extend(recovery_messages)
            
            # Now check the correct path
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": self.function_call_templates['list_dir'](path)
            })
        else:
            # Normal path check
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": self.function_call_templates['list_dir'](path)
            })
            
        # Function response shows directory exists
        messages.append({
            "role": "function",
            "name": "list_dir",
            "content": json.dumps({
                "status": "success",
                "data": {
                    "contents": [Path(path).name]
                }
            })
        })
        
        tools_used.append("list_dir")
        
        # Assistant response and next action
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('file_update', 1, True, {'path': path})
        })
        
        # Step 2: Read current content
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": self.function_call_templates['read_file'](path)
        })
        
        # Generate initial content based on type
        initial_content = random.choice(self.template_manager.file_templates[content_type]['basic']
                                      if content_type == 'python_script'
                                      else self.template_manager.file_templates[content_type])
        
        # Format template if needed
        if content_type == 'python_script' and 'description' in initial_content:
            initial_content = initial_content.format(description="An existing Python script")
        elif content_type == 'package_json':
            initial_content = initial_content.format(
                name=Path(path).stem,
                description="An existing package"
            )
        elif content_type == 'readme_md':
            initial_content = initial_content.format(
                project_name=Path(path).parent.name or "Project",
                description="An existing project",
                install_command="npm install" if content_type == 'package_json' else "pip install -r requirements.txt",
                usage_command="npm start" if content_type == 'package_json' else "python main.py",
                license="MIT"
            )
        
        messages.append({
            "role": "function",
            "name": "read_file",
            "content": json.dumps({
                "status": "success",
                "data": {
                    "content": initial_content
                }
            })
        })
        
        tools_used.append("read_file")
        
        # Assistant response and next action
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('file_update', 2, True, {'path': path})
        })
        
        # Step 3: Update content
        instruction, modified_content = self._generate_file_modifications(content_type, initial_content)
        
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": self.function_call_templates['edit_file'](
                path,
                modified_content,
                instruction
            )
        })
        
        messages.append({
            "role": "function",
            "name": "edit_file",
            "content": json.dumps(self.function_response_templates['edit_file']['success'](path))
        })
        
        tools_used.append("edit_file")
        
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('file_update', 3, True, {'path': path})
        })
        
        return {
            "messages": messages,
            "metadata": ScenarioMetadata(
                scenario_type="file_update",
                num_steps=len(tools_used),
                tools_used=tools_used,
                success=True
            ).to_dict()
        }
    
    def generate_project_setup_scenario(self, project_name: str = None, project_type: str = None) -> Dict[str, Any]:
        """Generate a project setup scenario with multiple steps"""
        if project_name is None:
            project_name = f"sample-project-{random.randint(1000, 9999)}"
        if project_type is None:
            project_type = random.choice(['python', 'nodejs'])
            
        messages = []
        tools_used = []
        
        # Step 1: Check if project directory exists
        messages.append({
            "role": "user",
            "content": f"Create a new {project_type} project called {project_name}"
        })
        
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": self.function_call_templates['list_dir'](project_name)
        })
        
        messages.append({
            "role": "function",
            "name": "list_dir",
            "content": json.dumps(self.function_response_templates['list_dir']['success'](project_name))
        })
        
        tools_used.append("list_dir")
        
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('project_setup', 1, True, {'project_name': project_name})
        })
        
        # Step 2: Create project directory
        mkdir_cmd = f"mkdir -p {project_name}"
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": self.function_call_templates['run_terminal_cmd'](mkdir_cmd)
        })
        
        messages.append({
            "role": "function",
            "name": "run_terminal_cmd",
            "content": json.dumps(self.function_response_templates['run_terminal_cmd']['success'](mkdir_cmd))
        })
        
        tools_used.append("run_terminal_cmd")
        
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('project_setup', 2, True, {'project_name': project_name})
        })
        
        # Step 3: Initialize git
        git_cmd = f"cd {project_name} && git init"
        messages.append({
            "role": "assistant",
            "content": "",
            "function_call": self.function_call_templates['run_terminal_cmd'](git_cmd)
        })
        
        messages.append({
            "role": "function",
            "name": "run_terminal_cmd",
            "content": json.dumps({
                "status": "success",
                "data": {
                    "stdout": "Initialized empty Git repository",
                    "stderr": ""
                }
            })
        })
        
        tools_used.append("run_terminal_cmd")
        
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('project_setup', 3, True, {'project_name': project_name})
        })
        
        # Get project structure
        structure = self._get_project_structure(project_type)
        
        # Step 4: Create project directories
        for directory in structure['directories']:
            dir_path = f"{project_name}/{directory}"
            mkdir_cmd = f"mkdir -p {dir_path}"
            
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": self.function_call_templates['run_terminal_cmd'](mkdir_cmd)
            })
            
            messages.append({
                "role": "function",
                "name": "run_terminal_cmd",
                "content": json.dumps(self.function_response_templates['run_terminal_cmd']['success'](mkdir_cmd))
            })
            
            tools_used.append("run_terminal_cmd")
        
        # Step 5: Create project files
        for file_path, content in structure['files'].items():
            full_path = f"{project_name}/{file_path}"
            
            # Format templates if needed
            if 'package.json' in file_path:
                content = content.format(
                    name=project_name,
                    description=f"A {project_type} project"
                )
            elif 'README.md' in file_path:
                content = content.format(
                    project_name=project_name,
                    description=f"A {project_type} project",
                    install_command="npm install" if project_type == 'nodejs' else "pip install -r requirements.txt",
                    usage_command="npm start" if project_type == 'nodejs' else "python src/main.py",
                    license="MIT"
                )
            
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": self.function_call_templates['edit_file'](
                    full_path,
                    content,
                    f"Create {file_path}"
                )
            })
            
            messages.append({
                "role": "function",
                "name": "edit_file",
                "content": json.dumps(self.function_response_templates['edit_file']['success'](full_path))
            })
            
            tools_used.append("edit_file")
            
            # Add appropriate response based on file type
            if 'package.json' in file_path or 'requirements.txt' in file_path:
                messages.append({
                    "role": "assistant",
                    "content": self._generate_assistant_response('project_setup', 4, True, {'project_name': project_name})
                })
            elif 'README.md' in file_path:
                messages.append({
                    "role": "assistant",
                    "content": self._generate_assistant_response('project_setup', 5, True, {'project_name': project_name})
                })
        
        # Final success message
        messages.append({
            "role": "assistant",
            "content": self._generate_assistant_response('project_setup', 6, True, {'project_name': project_name})
        })
        
        return {
            "messages": messages,
            "metadata": ScenarioMetadata(
                scenario_type="project_setup",
                num_steps=len(tools_used),
                tools_used=tools_used,
                success=True
            ).to_dict()
        }

class DatasetGenerator:
    """Main class for generating the multi-step tool usage dataset"""
    
    def __init__(self):
        self.template_manager = TemplateManager()
        self.scenario_generator = ScenarioGenerator(self.template_manager)
        self.error_handler = ErrorHandler()
        
        self.scenario_distribution = {
            'file_creation': 0.4,
            'file_update': 0.3,
            'project_setup': 0.2,
            'file_migration': 0.1
        }
        
        self.error_distribution = {
            'success': 0.7,
            'path_error': 0.1,
            'content_error': 0.1,
            'command_error': 0.1
        }
    
    def generate_dataset(self, num_examples: int) -> Dict[str, Any]:
        """Generate the complete dataset with the specified number of examples"""
        examples = []
        statistics = defaultdict(int)
        
        # Calculate number of examples for each scenario type
        scenario_counts = {}
        remaining = num_examples
        
        # Ensure at least one example of each type if we have enough examples
        if num_examples >= len(self.scenario_distribution):
            for scenario_type in self.scenario_distribution:
                scenario_counts[scenario_type] = 1
                remaining -= 1
        
        # Distribute remaining examples according to ratios
        if remaining > 0:
            for scenario_type, ratio in self.scenario_distribution.items():
                additional = int(remaining * ratio)
                scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + additional
        
        # Add any leftover examples to file_creation
        total_allocated = sum(scenario_counts.values())
        if total_allocated < num_examples:
            scenario_counts['file_creation'] = scenario_counts.get('file_creation', 0) + (num_examples - total_allocated)
        
        # Generate examples for each scenario type
        for scenario_type, count in scenario_counts.items():
            for _ in range(count):
                # Determine if this example should include an error
                should_error = random.random() > self.error_distribution['success']
                
                # Generate the scenario
                example = self._generate_scenario(scenario_type, should_error)
                examples.append(example)
                
                # Update statistics
                statistics['total'] += 1
                statistics[scenario_type] += 1
                if should_error:
                    statistics['errors'] += 1
        
        return {
            "data": examples,
            "statistics": dict(statistics)
        }
    
    def _generate_scenario(self, scenario_type: str, should_error: bool) -> Dict[str, Any]:
        """Generate a single scenario based on type and error flag"""
        if scenario_type == 'file_creation':
            return self.scenario_generator.generate_file_creation_scenario()
        elif scenario_type == 'file_update':
            return self.scenario_generator.generate_file_update_scenario()
        elif scenario_type == 'project_setup':
            return self.scenario_generator.generate_project_setup_scenario()
        else:  # file_migration
            # TODO: Implement file migration scenario
            return self.scenario_generator.generate_file_creation_scenario()  # Fallback for now

def main():
    parser = argparse.ArgumentParser(description='Generate expanded training data for multi-step tool usage')
    parser.add_argument('--num-examples', type=int, default=1000,
                      help='Number of examples to generate')
    parser.add_argument('--output-dir', type=str, default='generated_tool_dataset',
                      help='Output directory for generated data')
    parser.add_argument('--split-ratio', type=float, default=0.9,
                      help='Train/validation split ratio')
    
    args = parser.parse_args()
    
    # Use relative path from current working directory
    output_dir = Path(args.output_dir)
    
    logger.info(f"Output directory: {output_dir}")
    
    try:
        generator = DatasetGenerator()
        dataset = generator.generate_dataset(args.num_examples)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train and validation sets
        examples = dataset['data']
        split_idx = int(len(examples) * args.split_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Save datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_file = output_dir / f"train_{timestamp}.json"
        valid_file = output_dir / f"validation_{timestamp}.json"
        stats_file = output_dir / f"statistics_{timestamp}.json"
        
        logger.info(f"Saving training data to {train_file}")
        with open(train_file, 'w') as f:
            json.dump({"data": train_examples}, f, indent=2)
        
        logger.info(f"Saving validation data to {valid_file}")
        with open(valid_file, 'w') as f:
            json.dump({"data": val_examples}, f, indent=2)
        
        logger.info(f"Saving statistics to {stats_file}")
        with open(stats_file, 'w') as f:
            json.dump(dataset['statistics'], f, indent=2)
        
        logger.info(f"Generated {len(train_examples)} training and {len(val_examples)} validation examples")
        logger.info(f"Files saved in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 