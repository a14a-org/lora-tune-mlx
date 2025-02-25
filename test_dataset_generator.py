#!/usr/bin/env python3
import unittest
import json
from pathlib import Path
from generate_expanded_tool_dataset import (
    TemplateManager,
    ScenarioGenerator,
    ErrorHandler,
    DatasetGenerator
)

class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.template_manager = TemplateManager()
        self.scenario_generator = ScenarioGenerator(self.template_manager)
        self.error_handler = ErrorHandler()
        self.dataset_generator = DatasetGenerator()

    def validate_message_structure(self, messages):
        """Helper method to validate message structure"""
        self.assertTrue(len(messages) > 0)
        for message in messages:
            self.assertIn("role", message)
            self.assertIn("content", message)
            if message["role"] == "assistant" and "function_call" in message:
                self.assertIn("name", message["function_call"])
                self.assertIn("arguments", message["function_call"])

    def validate_scenario_metadata(self, metadata):
        """Helper method to validate scenario metadata"""
        self.assertIn("scenario_type", metadata)
        self.assertIn("num_steps", metadata)
        self.assertIn("tools_used", metadata)
        self.assertIn("success", metadata)
        self.assertIsInstance(metadata["tools_used"], list)
        self.assertGreater(len(metadata["tools_used"]), 0)

    def test_file_creation_scenario(self):
        """Test file creation scenario generation"""
        result = self.scenario_generator.generate_file_creation_scenario(
            path="src/test.py",
            content_type="python_script"
        )
        
        self.assertIn("messages", result)
        self.assertIn("metadata", result)
        
        messages = result["messages"]
        metadata = result["metadata"]
        
        self.validate_message_structure(messages)
        self.validate_scenario_metadata(metadata)
        
        self.assertEqual(metadata["scenario_type"], "file_creation")
        self.assertTrue(any("list_dir" in tools for tools in metadata["tools_used"]))
        self.assertTrue(any("edit_file" in tools for tools in metadata["tools_used"]))

    def test_file_update_scenario(self):
        """Test file update scenario generation"""
        result = self.scenario_generator.generate_file_update_scenario(
            path="src/test.py",
            content_type="python_script"
        )
        
        self.assertIn("messages", result)
        self.assertIn("metadata", result)
        
        messages = result["messages"]
        metadata = result["metadata"]
        
        self.validate_message_structure(messages)
        self.validate_scenario_metadata(metadata)
        
        self.assertEqual(metadata["scenario_type"], "file_update")
        self.assertTrue(any("read_file" in tools for tools in metadata["tools_used"]))
        self.assertTrue(any("edit_file" in tools for tools in metadata["tools_used"]))

    def test_project_setup_scenario(self):
        """Test project setup scenario generation"""
        result = self.scenario_generator.generate_project_setup_scenario(
            project_name="test-node-project",
            project_type="nodejs"
        )
        
        self.assertIn("messages", result)
        self.assertIn("metadata", result)
        
        messages = result["messages"]
        metadata = result["metadata"]
        
        self.validate_message_structure(messages)
        self.validate_scenario_metadata(metadata)
        
        self.assertEqual(metadata["scenario_type"], "project_setup")
        self.assertTrue(any("run_terminal_cmd" in tools for tools in metadata["tools_used"]))
        self.assertTrue(any("edit_file" in tools for tools in metadata["tools_used"]))

    def test_file_content_generation(self):
        """Test file content generation for different types"""
        # Test Python script generation
        result = self.scenario_generator.generate_file_creation_scenario(
            path="src/test.py",
            content_type="python_script"
        )
        messages = result["messages"]
        edit_message = next(m for m in messages if "function_call" in m and
                          m["function_call"]["name"] == "edit_file" and
                          json.loads(m["function_call"]["arguments"])["relative_workspace_path"].endswith(".py"))
        args = json.loads(edit_message["function_call"]["arguments"])
        self.assertIn("def", args["code_edit"])
        self.assertIn("main()", args["code_edit"])

        # Test package.json generation
        result = self.scenario_generator.generate_file_creation_scenario(
            path="package.json",
            content_type="package_json"
        )
        messages = result["messages"]
        edit_message = next(m for m in messages if "function_call" in m and
                          m["function_call"]["name"] == "edit_file" and
                          json.loads(m["function_call"]["arguments"])["relative_workspace_path"].endswith("package.json"))
        args = json.loads(edit_message["function_call"]["arguments"])
        content = json.loads(args["code_edit"])
        self.assertIn("name", content)
        self.assertIn("version", content)
        self.assertIn("scripts", content)

    def test_error_handling(self):
        """Test error handling in the error handler"""
        error_params = {"path": "nonexistent/file.py"}
        
        # Test path not found error
        error_response = self.error_handler.generate_error_response("path_not_found", error_params)
        self.assertEqual(error_response["status"], "error")
        self.assertIn("type", error_response["error"])
        self.assertIn("message", error_response["error"])
        self.assertEqual(error_response["error"]["type"], "path_not_found")
        self.assertIn("nonexistent/file.py", error_response["error"]["message"])

if __name__ == '__main__':
    unittest.main() 