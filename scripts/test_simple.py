import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_lm import load, generate
import mlx.core as mx
from models import LoRALinear
from pathlib import Path
import json
import argparse

# Load the model
print("Loading model...")
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-bf16")

# Load LoRA config
print("Loading LoRA config...")
with open("config/lora_qwen_config.json", "r") as f:
    config = json.load(f)
lora_config = config["lora_config"]

# Apply LoRA layers
print("Applying LoRA layers...")
transformer = model.model
if hasattr(transformer, "layers"):
    layers = transformer.layers
elif hasattr(transformer, "h"):
    layers = transformer.h
else:
    raise ValueError("Could not find layers in model structure")

print("\nApplying LoRA to layers...")
for layer in layers:
    # Get the original Q and V projections
    if "q_proj" in lora_config["target_modules"]:
        q_proj = layer.self_attn.q_proj
        # Check if original layer has bias
        has_bias = hasattr(q_proj, "bias") and q_proj.bias is not None
        q_lora = LoRALinear(
            q_proj.weight.shape[1],
            q_proj.weight.shape[0],
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            bias=has_bias
        )
        # Copy weights and bias
        q_lora.weight = q_proj.weight
        if has_bias:
            q_lora.bias = q_proj.bias
        layer.self_attn.q_proj = q_lora
        print(f"Applied LoRA to q_proj in layer")
    
    if "v_proj" in lora_config["target_modules"]:
        v_proj = layer.self_attn.v_proj
        has_bias = hasattr(v_proj, "bias") and v_proj.bias is not None
        v_lora = LoRALinear(
            v_proj.weight.shape[1],
            v_proj.weight.shape[0],
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            bias=has_bias
        )
        v_lora.weight = v_proj.weight
        if has_bias:
            v_lora.bias = v_proj.bias
        layer.self_attn.v_proj = v_lora
        print(f"Applied LoRA to v_proj in layer")

# Load LoRA weights
print("\nLoading LoRA weights...")
checkpoint_path = Path("lora_checkpoints/checkpoint-2000.npz")
if checkpoint_path.exists():
    print(f"Found checkpoint at {checkpoint_path}")
    lora_state = mx.load(str(checkpoint_path))
    print(f"Loaded state dict with {len(lora_state)} entries")
    
    # Apply weights to LoRA layers
    lora_params_loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_state:
                module.lora_A = lora_state[f"{name}.lora_A"]
                lora_params_loaded += 1
            if f"{name}.lora_B" in lora_state:
                module.lora_B = lora_state[f"{name}.lora_B"]
                lora_params_loaded += 1
    print(f"Successfully loaded {lora_params_loaded} LoRA parameters")
else:
    print("No LoRA weights found at", checkpoint_path)

def get_enhanced_system_prompt(include_zero_shot_tool=False):
    """Get the enhanced system prompt with tool definitions and examples."""
    base_prompt = """<|im_start|>system
You are a powerful agentic AI coding assistant. You help users with coding tasks using the following tools:

<tool_definition name='list_dir'>
  description: Lists directory contents at a specified path relative to the workspace
  parameters:
    - relative_workspace_path (string, required): Path to list contents of
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
</tool_definition>

<tool_definition name='read_file'>
  description: Reads file contents, with support for both full file and partial reading
  parameters:
    - relative_workspace_path (string, required): Path to the file
    - should_read_entire_file (boolean, required): Whether to read entire file
    - start_line_one_indexed (integer, required): Start line (1-based)
    - end_line_one_indexed_inclusive (integer, required): End line (1-based)
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
</tool_definition>

<tool_definition name='edit_file'>
  description: Makes code changes to specified files based on instructions
  parameters:
    - target_file (string, required): File to edit
    - instructions (string, required): What changes to make
    - code_edit (string, required): The actual edit content
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
</tool_definition>

<tool_definition name='run_terminal_cmd'>
  description: Executes terminal commands with configurable execution options
  parameters:
    - command (string, required): Command to execute
    - is_background (boolean, required): Whether to run in background
    - require_user_approval (boolean, required): Whether user must approve
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
</tool_definition>"""

    # Add zero-shot web_request tool if requested
    if include_zero_shot_tool:
        base_prompt += """

<tool_definition name='web_request'>
  description: Makes HTTP requests to retrieve information from the web
  parameters:
    - url (string, required): The URL to make the request to
    - method (string, required): HTTP method (GET, POST, etc.)
    - headers (string, optional): JSON string of headers to include
    - data (string, optional): Data to send in the request body
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
</tool_definition>"""

    # Add examples and closing
    base_prompt += """

Example conversations:

1. File creation and editing:
User: Can you create a new Python script that prints "Hello, World!"?
Assistant: I'll help you create a new Python script.
<tool name='edit_file'>target_file="hello.py" instructions="Create a new Python script" code_edit="#!/usr/bin/env python3\n\nprint('Hello, World!')"</tool>

2. File reading:
User: What's in the requirements.txt file?
Assistant: I'll check the contents of requirements.txt for you.
<tool name='read_file'>relative_workspace_path="requirements.txt" should_read_entire_file=true start_line_one_indexed=1 end_line_one_indexed_inclusive=100</tool>

Remember:
1. Only use the tools defined above
2. Always use proper XML-style tags for tool calls
3. Do not include or simulate system responses
4. Wait for actual tool responses before proceeding
<|im_end|>"""

    return base_prompt

# Test prompts
def get_test_prompts(include_zero_shot_tool=False):
    system_prompt = get_enhanced_system_prompt(include_zero_shot_tool)
    
    standard_prompts = [
        system_prompt + """<|im_start|>user
Can you check what files are in the current directory?
<|im_end|>
<|im_start|>assistant
""",
        system_prompt + """<|im_start|>user
Create a new file called test.py with a simple unit test example.
<|im_end|>
<|im_start|>assistant
""",
        system_prompt + """<|im_start|>user
What's in the requirements.txt file?
<|im_end|>
<|im_start|>assistant
"""
    ]
    
    # Add zero-shot tool test if requested
    if include_zero_shot_tool:
        zero_shot_prompt = system_prompt + """<|im_start|>user
Can you check the current weather in San Francisco?
<|im_end|>
<|im_start|>assistant
"""
        standard_prompts.append(zero_shot_prompt)
    
    return standard_prompts

def get_multi_step_prompts():
    """Get prompts for multi-step conversation testing with tool responses."""
    base_prompt = get_enhanced_system_prompt()
    
    file_creation_convo = base_prompt + """<|im_start|>user
Can you create a new Python script that calculates fibonacci numbers?
<|im_end|>
<|im_start|>assistant
I'll help you create a Python script for calculating Fibonacci numbers.
<tool name='edit_file'>target_file="fibonacci.py" instructions="Create a new Python script for Fibonacci calculation" code_edit="def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    n = 10
    print(f'Fibonacci({n}) = {fibonacci(n)}')

if __name__ == '__main__':
    main()"</tool>
<|im_end|>
<|im_start|>system
<tool_response name='edit_file'>{"status":"success","message":"File created successfully"}</tool_response>
<|im_end|>
<|im_start|>user
Can you add a docstring to explain the function?
<|im_end|>
<|im_start|>assistant
"""

    project_setup_convo = base_prompt + """<|im_start|>user
Can you help me set up a new Python project with proper structure?
<|im_end|>
<|im_start|>assistant
I'll help you create a proper Python project structure. First, let's create the necessary directories.
<tool name='run_terminal_cmd'>command="mkdir -p src tests docs" is_background=false require_user_approval=true</tool>
<|im_end|>
<|im_start|>system
<tool_response name='run_terminal_cmd'>{"status":"success","stdout":"","stderr":""}</tool_response>
<|im_end|>
<|im_start|>user
Great! Can you create a basic setup.py file?
<|im_end|>
<|im_start|>assistant
"""

    return [file_creation_convo, project_setup_convo]

def run_single_step_tests(model, tokenizer, include_zero_shot=False):
    """Run the original single-step test cases."""
    print("\nRunning single-step tests:")
    print("=" * 50)

    for i, prompt in enumerate(test_prompts):
        parts = prompt.split("<|im_start|>")
        for part in parts:
            if part.startswith("user\n"):
                question = part.split("\n")[1].split("<|im_end|>")[0]
                break

        # Add special label for zero-shot test
        test_label = f"Test Case {i + 1}"
        if include_zero_shot and i == len(test_prompts) - 1:
            test_label = "Zero-Shot Tool Test"

        print(f"\n{test_label}:")
        print(f"Prompt: {question}")
        print("\nGenerating response...")
        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        print(f"\nResponse: {response}")
        print("=" * 50)

def run_multi_step_tests(model, tokenizer):
    """Run multi-step conversation tests with tool responses."""
    print("\nRunning multi-step conversation tests:")
    print("=" * 80)

    multi_step_prompts = get_multi_step_prompts()
    scenarios = [
        "File creation and editing",
        "Project setup"
    ]

    for i, (prompt, scenario) in enumerate(zip(multi_step_prompts, scenarios)):
        print(f"\nMulti-step Test {i + 1}: {scenario}")
        print("=" * 80)
        
        # Extract and display the conversation history
        parts = prompt.split("<|im_start|>")
        conversation = []
        for part in parts:
            if part.strip() == "":
                continue
            role = part.split("\n")[0]
            content = "\n".join(part.split("\n")[1:]).split("<|im_end|>")[0].strip()
            if role != "system" or "<tool_response" in content:  # Skip system prompt unless it's a tool response
                conversation.append((role, content))
        
        # Display conversation history
        print("\nConversation history:")
        print("-" * 40)
        for role, content in conversation:
            if "<tool_response" in content:
                print(f"\n🔧 Tool Response:")
                print(f"{content}")
            elif role == "user":
                print(f"\n👤 User:")
                print(f"{content}")
            elif role == "assistant":
                print(f"\n🤖 Assistant:")
                print(f"{content}")
        
        # Generate and display new response
        print(f"\n🤖 Assistant (New Response):")
        print("\nGenerating response...")
        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        print(f"\nResponse: {response}")
        print("=" * 80)

def run_interactive_multi_step_test(model, tokenizer, scenario, include_failure=False):
    """Run multi-step conversation test interactively, one step at a time."""
    print("\nRunning interactive multi-step test:")
    print("=" * 80)
    
    base_prompt = get_enhanced_system_prompt()
    
    if scenario == "file_creation":
        # File creation scenario
        initial_prompt = base_prompt + """<|im_start|>user
Can you create a new Python script that calculates fibonacci numbers?
<|im_end|>
<|im_start|>assistant
"""
        tool_response = '<tool_response name="edit_file">{"status":"success","message":"File created successfully"}</tool_response>' if include_failure else '<tool_response name="edit_file">{"status":"success","message":"File created successfully"}</tool_response>'
        
    elif scenario == "project_setup":
        # Project setup scenario
        initial_prompt = base_prompt + """<|im_start|>user
Can you help me set up a new Python project with proper structure?
<|im_end|>
<|im_start|>assistant
"""
        tool_response = '<tool_response name="run_terminal_cmd">{"status":"success","stdout":"","stderr":""}</tool_response>'
    
    # Step 1: Initial user request
    print("\nStep 1: Initial user request")
    print("-" * 40)
    print("\n👤 User:")
    print(initial_prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
    
    print("\n🤖 Assistant (Generating response...):")
    response1 = generate(model, tokenizer, prompt=initial_prompt, verbose=True)
    print(f"\nResponse: {response1}")
    
    # Step 2: Add tool response and get next assistant response
    input("\nPress Enter to continue with tool response...")
    
    extended_prompt = base_prompt + """<|im_start|>user
""" + initial_prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] + """
<|im_end|>
<|im_start|>assistant
""" + response1 + """
<|im_end|>
<|im_start|>system
""" + tool_response + """
<|im_end|>
<|im_start|>assistant
"""
    
    print("\nStep 2: Tool response")
    print("-" * 40)
    print("\n🔧 Tool Response:")
    print(tool_response)
    
    print("\n🤖 Assistant (Generating response...):")
    response2 = generate(model, tokenizer, prompt=extended_prompt, verbose=True)
    print(f"\nResponse: {response2}")
    
    if scenario == "project_setup":
        # Add project setup step for project setup scenario
        input("\nPress Enter to continue with project setup...")
        
        extended_prompt += response2 + """
<|im_end|>
<|im_start|>user
Great! Can you create a basic setup.py file?
<|im_end|>
<|im_start|>assistant
"""
        
        print("\nStep 3: User requests project setup")
        print("-" * 40)
        print("\n👤 User:")
        print("Great! Can you create a basic setup.py file?")
        
        print("\n🤖 Assistant (Generating response...):")
        response3 = generate(model, tokenizer, prompt=extended_prompt, verbose=True)
        print(f"\nResponse: {response3}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Run model inference tests')
    parser.add_argument('--multi-step-test', action='store_true',
                      help='Run multi-step conversation tests')
    parser.add_argument('--interactive', action='store_true',
                      help='Run interactive multi-step test')
    parser.add_argument('--file-creation', action='store_true',
                      help='Run file creation scenario')
    parser.add_argument('--project-setup', action='store_true',
                      help='Run project setup scenario')
    parser.add_argument('--failure', action='store_true',
                      help='Include failure cases in tests')
    parser.add_argument('--checkpoint-path', type=str, default='lora_checkpoints/checkpoint-2000.npz',
                      help='Path to the checkpoint file')
    parser.add_argument('--zero-shot', action='store_true',
                      help='Include zero-shot tool test with web_request tool')
    args = parser.parse_args()

    # Update checkpoint path based on argument
    global checkpoint_path
    checkpoint_path = Path(args.checkpoint_path)

    # Load model and apply LoRA (existing code)
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-bf16")

    # Load LoRA config and apply layers (existing code)
    print("Loading LoRA config...")
    with open("config/lora_qwen_config.json", "r") as f:
        config = json.load(f)
    lora_config = config["lora_config"]

    # Apply LoRA layers (existing code)
    print("Applying LoRA layers...")
    transformer = model.model
    if hasattr(transformer, "layers"):
        layers = transformer.layers
    elif hasattr(transformer, "h"):
        layers = transformer.h
    else:
        raise ValueError("Could not find layers in model structure")

    print("\nApplying LoRA to layers...")
    for layer in layers:
        if "q_proj" in lora_config["target_modules"]:
            q_proj = layer.self_attn.q_proj
            has_bias = hasattr(q_proj, "bias") and q_proj.bias is not None
            q_lora = LoRALinear(
                q_proj.weight.shape[1],
                q_proj.weight.shape[0],
                r=lora_config["r"],
                alpha=lora_config["alpha"],
                dropout=lora_config["dropout"],
                bias=has_bias
            )
            q_lora.weight = q_proj.weight
            if has_bias:
                q_lora.bias = q_proj.bias
            layer.self_attn.q_proj = q_lora
            print(f"Applied LoRA to q_proj in layer")
        
        if "v_proj" in lora_config["target_modules"]:
            v_proj = layer.self_attn.v_proj
            has_bias = hasattr(v_proj, "bias") and v_proj.bias is not None
            v_lora = LoRALinear(
                v_proj.weight.shape[1],
                v_proj.weight.shape[0],
                r=lora_config["r"],
                alpha=lora_config["alpha"],
                dropout=lora_config["dropout"],
                bias=has_bias
            )
            v_lora.weight = v_proj.weight
            if has_bias:
                v_lora.bias = v_proj.bias
            layer.self_attn.v_proj = v_lora
            print(f"Applied LoRA to v_proj in layer")

    # Load LoRA weights (existing code)
    print("\nLoading LoRA weights...")
    checkpoint_path = Path("lora_checkpoints/checkpoint-2000.npz")
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}")
        lora_state = mx.load(str(checkpoint_path))
        print(f"Loaded state dict with {len(lora_state)} entries")
        
        lora_params_loaded = 0
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state:
                    module.lora_A = lora_state[f"{name}.lora_A"]
                    lora_params_loaded += 1
                if f"{name}.lora_B" in lora_state:
                    module.lora_B = lora_state[f"{name}.lora_B"]
                    lora_params_loaded += 1
        print(f"Successfully loaded {lora_params_loaded} LoRA parameters")
    else:
        print("No LoRA weights found at", checkpoint_path)

    # Get test prompts with or without zero-shot tool
    global test_prompts
    test_prompts = get_test_prompts(args.zero_shot)

    # Run tests based on command line arguments
    if args.file_creation:
        run_interactive_multi_step_test(model, tokenizer, "file_creation", args.failure)
    if args.project_setup:
        run_interactive_multi_step_test(model, tokenizer, "project_setup", args.failure)
    if not (args.file_creation or args.project_setup):  # Default behavior
        if args.interactive:
            run_interactive_multi_step_test(model, tokenizer, "file_creation", args.failure)
        elif args.multi_step_test:
            run_multi_step_tests(model, tokenizer)
        else:
            run_single_step_tests(model, tokenizer, args.zero_shot)

if __name__ == "__main__":
    main() 