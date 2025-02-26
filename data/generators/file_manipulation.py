"""Generator for file manipulation conversations."""

import random
import os
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseGenerator
from data.templates.constants import (
    PYTHON_FILES,
    COMMON_FUNCTIONS,
    FUNCTION_TEMPLATES,
    ERROR_PATTERNS,
    IMPROVEMENT_TYPES,
    PROJECT_TEMPLATES,
    TEST_TEMPLATES
)

class FileManipulationGenerator(BaseGenerator):
    """Generator for file manipulation conversations."""
    
    def __init__(self):
        super().__init__()
        self.project_state = {
            'files': set(),  # Track created files
            'functions': {},  # Track function implementations
            'dependencies': {}  # Track file dependencies
        }
    
    def generate(self) -> Dict[str, Any]:
        """Generate a conversation about file manipulation."""
        # Reset messages for new conversation
        self.messages = []
        
        # Select scenario and template
        scenario = self._select_scenario()
        return self._generate_scenario(scenario)
    
    def _select_scenario(self) -> str:
        """Select a conversation scenario based on weights."""
        scenarios = [
            ('new_function', 0.4),
            ('improve_existing', 0.3),
            ('fix_bug', 0.2),
            ('add_tests', 0.1)
        ]
        return random.choices(
            [s[0] for s in scenarios],
            weights=[s[1] for s in scenarios]
        )[0]
    
    def _select_template(self) -> Tuple[str, Dict]:
        """Select a function template category and specific template."""
        category = random.choice(list(FUNCTION_TEMPLATES.keys()))
        complexity = random.choice(list(FUNCTION_TEMPLATES[category].keys()))
        return category, FUNCTION_TEMPLATES[category][complexity]
    
    def _generate_scenario(self, scenario: str) -> Dict[str, Any]:
        """Generate conversation based on selected scenario."""
        if scenario == 'new_function':
            return self._generate_new_function()
        elif scenario == 'improve_existing':
            return self._generate_improvement()
        elif scenario == 'fix_bug':
            return self._generate_bug_fix()
        else:  # add_tests
            return self._generate_add_tests()
    
    def _generate_new_function(self) -> Dict[str, Any]:
        """Generate conversation about creating a new function."""
        # Select file and function names
        filename = random.choice(PYTHON_FILES)
        function_name = random.choice(COMMON_FUNCTIONS)
        
        # Select template
        category, template_data = self._select_template()
        
        # Get base template parameters
        template_params = self._get_base_template_params(
            function_name,
            f"Implement {function_name} functionality"
        )
        
        # Update parameters based on category
        if category == 'data_processing':
            if 'operations' in template_data:
                operation = random.choice(list(template_data['operations'].keys()))
                template_params.update({
                    'operation': operation,
                    'operation_impl': template_data['operations'][operation]
                })
        elif category == 'file_operations':
            if 'operations' in template_data:
                operation = random.choice(list(template_data['operations'].keys()))
                op_info = template_data['operations'][operation]
                template_params.update({
                    'return_type': op_info['return_type'],
                    'operation_desc': op_info['desc'],
                    'return_desc': op_info['return_desc'],
                    'operation_impl': op_info['impl']
                })
        
        # Generate implementation
        implementation = template_data['template'].format(**template_params)
        
        # Add conversation messages
        self.add_user_message(f"Can you create a Python file that implements {function_name}?")
        
        # Create file
        self.add_tool_call(
            "edit_file",
            {
                "target_file": filename,
                "instructions": f"Create {filename} with {function_name} implementation",
                "code_edit": implementation
            },
            {"success": True, "changes": [{"type": "create", "path": filename}]}
        )
        
        # Update project state
        self.project_state['files'].add(filename)
        self.project_state['functions'][function_name] = {
            'file': filename,
            'category': category,
            'template': template_data
        }
        
        # Add assistant response
        self.add_assistant_message(
            f"I've created {filename} with an implementation of {function_name} "
            f"using a {category} pattern. The function includes basic error handling "
            "and documentation. Would you like me to add any specific improvements?"
        )
        
        return self.get_conversation()
    
    def _generate_improvement(self) -> Dict[str, Any]:
        """Generate conversation about improving an existing function."""
        if not self.project_state['functions']:
            return self._generate_new_function()
        
        # Select function to improve
        function_name = random.choice(list(self.project_state['functions'].keys()))
        function_info = self.project_state['functions'][function_name]
        filename = function_info['file']
        
        # Select improvement type
        improvement = random.choice(list(IMPROVEMENT_TYPES.keys()))
        
        # Generate improvement request
        self.add_user_message(
            f"Can you improve the {function_name} function with better {improvement}?"
        )
        
        # Apply improvements based on type
        if improvement == 'error_handling':
            new_content = self._add_error_handling(function_name, function_info)
        elif improvement == 'documentation':
            new_content = self._add_documentation(function_name, function_info)
        elif improvement == 'optimization':
            new_content = self._add_optimization(function_name, function_info)
        else:  # testing
            return self._generate_add_tests()
        
        # Apply improvements
        self.add_tool_call(
            "edit_file",
            {
                "target_file": filename,
                "instructions": f"Improve {function_name} with better {improvement}",
                "code_edit": new_content
            },
            {"success": True, "changes": [{"type": "modify", "path": filename}]}
        )
        
        # Add assistant response
        self.add_assistant_message(
            f"I've updated {filename} to improve {improvement} in the {function_name} function. "
            "The changes include:\n"
            f"- {IMPROVEMENT_TYPES[improvement]['description']}\n"
            "Would you like me to make any other improvements?"
        )
        
        return self.get_conversation()
    
    def _generate_bug_fix(self) -> Dict[str, Any]:
        """Generate conversation about fixing a bug."""
        if not self.project_state['functions']:
            return self._generate_new_function()
        
        # Select function to fix
        function_name = random.choice(list(self.project_state['functions'].keys()))
        function_info = self.project_state['functions'][function_name]
        filename = function_info['file']
        
        # Generate bug description
        bug_types = [
            "edge case handling",
            "type conversion",
            "null checking",
            "boundary conditions"
        ]
        bug_type = random.choice(bug_types)
        
        # Add bug report
        self.add_user_message(
            f"I found a bug in the {function_name} function related to {bug_type}. "
            "Can you fix it?"
        )
        
        # Generate fix
        new_content = self._generate_bug_fix_implementation(
            function_name,
            function_info,
            bug_type
        )
        
        # Apply fix
        self.add_tool_call(
            "edit_file",
            {
                "target_file": filename,
                "instructions": f"Fix {bug_type} bug in {function_name}",
                "code_edit": new_content
            },
            {"success": True, "changes": [{"type": "modify", "path": filename}]}
        )
        
        # Add assistant response
        self.add_assistant_message(
            f"I've fixed the {bug_type} issue in the {function_name} function. "
            "The changes include:\n"
            "- Additional validation\n"
            "- Improved error messages\n"
            "- Edge case handling\n"
            "Would you like me to add tests to prevent similar bugs?"
        )
        
        return self.get_conversation()
    
    def _generate_add_tests(self) -> Dict[str, Any]:
        """Generate conversation about adding tests."""
        if not self.project_state['functions']:
            return self._generate_new_function()
        
        # Select function to test
        function_name = random.choice(list(self.project_state['functions'].keys()))
        function_info = self.project_state['functions'][function_name]
        
        # Create test file
        test_filename = f"test_{function_info['file']}"
        
        # Generate test cases
        test_content = self._generate_test_cases(function_name, function_info)
        
        # Add test request
        self.add_user_message(
            f"Can you add tests for the {function_name} function?"
        )
        
        # Create test file
        self.add_tool_call(
            "edit_file",
            {
                "target_file": test_filename,
                "instructions": f"Create tests for {function_name}",
                "code_edit": test_content
            },
            {"success": True, "changes": [{"type": "create", "path": test_filename}]}
        )
        
        # Update project state
        self.project_state['files'].add(test_filename)
        
        # Add assistant response
        self.add_assistant_message(
            f"I've created {test_filename} with comprehensive tests for {function_name}. "
            "The tests include:\n"
            "- Basic functionality tests\n"
            "- Edge case tests\n"
            "- Error handling tests\n"
            "Would you like me to add any specific test cases?"
        )
        
        return self.get_conversation()
    
    def _get_base_template_params(self, function_name: str, description: str) -> Dict[str, Any]:
        """Get base template parameters that are common across all methods."""
        return {
            'name': function_name,
            'operation_desc': description,
            'return_type': 'Any',
            'return_desc': 'Processed result',
            'operation_impl': 'pass',
            'method': 'get',  # Default for API handling
            'filepath': 'input_file.txt',  # Default for file operations
            'operation': 'process'  # Default operation name
        }
    
    def _add_error_handling(self, function_name: str, function_info: Dict) -> str:
        """Add error handling to a function."""
        category = function_info['category']
        template = function_info['template']
        
        # Select error patterns based on category
        if category == 'data_processing':
            patterns = [
                ERROR_PATTERNS['type_check']['multiple'],
                ERROR_PATTERNS['value_check']['non_empty']
            ]
        elif category == 'file_operations':
            patterns = [
                ERROR_PATTERNS['file_check']['exists'],
                ERROR_PATTERNS['file_check']['is_file']
            ]
        else:
            patterns = [
                ERROR_PATTERNS['type_check']['single'],
                ERROR_PATTERNS['value_check']['range']
            ]
        
        # Get base parameters and update with error handling specifics
        template_params = self._get_base_template_params(
            function_name,
            f"Enhanced {function_name} with error handling"
        )
        template_params['operation_impl'] = '\n    '.join(patterns)
        
        # Add error handling to template
        return template['template'].format(**template_params)
    
    def _add_documentation(self, function_name: str, function_info: Dict) -> str:
        """Add improved documentation to a function."""
        category = function_info['category']
        template = function_info['template']
        
        # Add detailed docstring sections
        docstring = []
        docstring.append(f"{function_name}: {IMPROVEMENT_TYPES['documentation']['description']}")
        docstring.append("")
        
        for section in IMPROVEMENT_TYPES['documentation']['sections']:
            docstring.append(f"{section}:")
            if section == 'Args':
                docstring.append("    data: Input data to process")
            elif section == 'Returns':
                docstring.append("    Processed and validated result")
            elif section == 'Raises':
                docstring.append("    ValueError: If input validation fails")
                docstring.append("    TypeError: If input type is incorrect")
            elif section == 'Examples':
                docstring.append("    >>> result = process_data([1, 2, 3])")
                docstring.append("    >>> print(result)")
            docstring.append("")
        
        # Get base parameters and update with documentation specifics
        template_params = self._get_base_template_params(
            function_name,
            '\n    '.join(docstring)
        )
        template_params['operation_impl'] = 'return data'
        
        # Update template with new documentation
        return template['template'].format(**template_params)
    
    def _add_optimization(self, function_name: str, function_info: Dict) -> str:
        """Add optimizations to a function."""
        category = function_info['category']
        template = function_info['template']
        
        # Select optimization technique
        technique = random.choice(IMPROVEMENT_TYPES['optimization']['techniques'])
        
        # Apply optimization based on technique
        if technique == 'caching':
            optimization = "@functools.lru_cache(maxsize=128)\n"
            impl = "return cached_result"
        elif technique == 'algorithm_improvement':
            optimization = "# Using optimized algorithm\n"
            impl = "return optimized_result"
        else:  # data_structure_optimization
            optimization = "# Using optimized data structure\n"
            impl = "return processed_data"
        
        # Get base parameters and update with optimization specifics
        template_params = self._get_base_template_params(
            function_name,
            f"Optimized {function_name} using {technique}"
        )
        template_params['operation_impl'] = impl
        
        # Update template with optimization
        return optimization + template['template'].format(**template_params)
    
    def _generate_bug_fix_implementation(
        self,
        function_name: str,
        function_info: Dict,
        bug_type: str
    ) -> str:
        """Generate bug fix implementation."""
        category = function_info['category']
        template = function_info['template']
        
        # Add fix based on bug type
        if bug_type == 'edge case handling':
            fix = ERROR_PATTERNS['value_check']['range']
        elif bug_type == 'type conversion':
            fix = ERROR_PATTERNS['type_check']['multiple']
        elif bug_type == 'null checking':
            fix = ERROR_PATTERNS['value_check']['non_empty']
        else:  # boundary conditions
            fix = ERROR_PATTERNS['value_check']['positive']
        
        # Get base parameters and update with bug fix specifics
        template_params = self._get_base_template_params(
            function_name,
            f"Fixed {bug_type} in {function_name}"
        )
        template_params['operation_impl'] = fix
        
        # Update template with fix
        return template['template'].format(**template_params)
    
    def _generate_test_cases(self, function_name: str, function_info: Dict) -> str:
        """Generate test cases for a function."""
        category = function_info['category']
        
        # Generate test file content
        content = 'import pytest\n\n'
        
        # Add basic test
        content += TEST_TEMPLATES['unit_test'].format(
            function_name=function_name,
            scenario='basic',
            setup='    input_data = [1, 2, 3]',
            assertion='    assert function_name(input_data) is not None'
        )
        
        # Add error test
        content += '\n\n' + TEST_TEMPLATES['exception_test'].format(
            function_name=function_name,
            error_type='ValueError',
            test_code=f'    {function_name}([])'
        )
        
        return content 