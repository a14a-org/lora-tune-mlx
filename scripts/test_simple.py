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
checkpoint_path = Path("lora_checkpoints/checkpoint-500.npz")
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

def get_enhanced_system_prompt():
    """Get the enhanced system prompt with tool definitions and examples."""
    return """<|im_start|>system
[SYSTEM VERSION 1.3]
You are a home automation assistant. You must use the exact tool call format shown in the examples below.

Available tools and usage examples:

<tool_definition name='control_lights'>
  description: Control smart lights in a room
  parameters:
    - room (string): The room where the lights are located
    - action (string, values: ['on', 'off', 'dim']): The action to perform on the lights
    - brightness (integer, optional): Brightness level (0-100), required for dim action
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
  examples:
    - User: Turn on the living room lights
      Assistant: I'll help you with that.
      <tool name='control_lights'>room="living room" action="on"</tool>
      System: <tool_response name='control_lights'>{"status":"success","message":"Lights turned on"}</tool_response>
      Assistant: The living room lights have been turned on.
</tool_definition>

<tool_definition name='get_weather'>
  description: Get current weather for a location
  parameters:
    - location (string): The city or location to get weather for
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
  examples:
    - User: What's the weather in Amsterdam?
      Assistant: I'll check the weather in Amsterdam for you.
      <tool name='get_weather'>location="Amsterdam"</tool>
      System: <tool_response name='get_weather'>{"temperature":18,"conditions":"sunny"}</tool_response>
      Assistant: In Amsterdam, it's currently 18°C and sunny.
</tool_definition>

<tool_definition name='set_thermostat'>
  description: Set thermostat temperature
  parameters:
    - temperature (number): The target temperature
    - unit (string, optional, values: ['C', 'F']): Temperature unit (Celsius or Fahrenheit)
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
  examples:
    - User: Set the temperature to 22 degrees Celsius
      Assistant: I'll set the thermostat to 22°C.
      <tool name='set_thermostat'>temperature=22 unit="C"</tool>
      System: <tool_response name='set_thermostat'>{"status":"success","set_temp":22,"unit":"C"}</tool_response>
      Assistant: The thermostat has been set to 22°C.
</tool_definition>

<tool_definition name='set_thermostat_delta'>
  description: Adjust temperature relative to current setting
  parameters:
    - delta (number): The temperature change (+/- degrees)
  format_rules:
    - Tool calls must use XML-style tags
    - Parameters must be space-separated key=value pairs
    - String values must be quoted
  examples:
    - User: Make it 2 degrees warmer
      Assistant: I'll increase the temperature by 2 degrees.
      <tool name='set_thermostat_delta'>delta=2</tool>
      System: <tool_response name='set_thermostat_delta'>{"status":"success","delta":2,"new_temp":24}</tool_response>
      Assistant: I've increased the temperature by 2 degrees.
</tool_definition>

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
<|im_end|>"""

# Test prompts
test_prompts = [
    get_enhanced_system_prompt() + """<|im_start|>user
Can you turn on the lights in the living room?
<|im_end|>
<|im_start|>assistant
""",
    get_enhanced_system_prompt() + """<|im_start|>user
What's the current temperature in the house?
<|im_end|>
<|im_start|>assistant
""",
    get_enhanced_system_prompt() + """<|im_start|>user
Can you check the weather in Amsterdam?
<|im_end|>
<|im_start|>assistant
"""
]

def get_multi_step_prompts():
    """Get prompts for multi-step conversation testing with tool responses."""
    base_prompt = get_enhanced_system_prompt()
    
    weather_temp_convo = base_prompt + """<|im_start|>user
What's the weather like in London?
<|im_end|>
<|im_start|>assistant
I'll check the weather in London for you.
<tool name='get_weather'>location="London"</tool>
<|im_end|>
<|im_start|>system
<tool_response name='get_weather'>{"temperature":15,"conditions":"cloudy"}</tool_response>
<|im_end|>
<|im_start|>assistant
In London, it's currently 15°C and cloudy. Would you like me to adjust the thermostat to match the outdoor temperature?
<|im_end|>
<|im_start|>user
Yes, please set it to 20 degrees
<|im_end|>
<|im_start|>assistant
"""

    lights_feedback_convo = base_prompt + """<|im_start|>user
Can you turn on the kitchen lights?
<|im_end|>
<|im_start|>assistant
I'll help you turn on the kitchen lights.
<tool name='control_lights'>room="kitchen" action="on"</tool>
<|im_end|>
<|im_start|>system
<tool_response name='control_lights'>{"status":"error","message":"Kitchen lights not responding"}</tool_response>
<|im_end|>
<|im_start|>assistant
"""

    return [weather_temp_convo, lights_feedback_convo]

def run_single_step_tests(model, tokenizer):
    """Run the original single-step test cases."""
    print("\nRunning single-step tests:")
    print("=" * 50)

    for i, prompt in enumerate(test_prompts):
        parts = prompt.split("<|im_start|>")
        for part in parts:
            if part.startswith("user\n"):
                question = part.split("\n")[1].split("<|im_end|>")[0]
                break

        print(f"\nTest Case {i + 1}:")
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
        "Weather query with temperature adjustment suggestion",
        "Light control with error handling"
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
    
    if scenario == "lights":
        # Lights scenario
        initial_prompt = base_prompt + """<|im_start|>user
Can you turn on the kitchen lights?
<|im_end|>
<|im_start|>assistant
"""
        tool_response = '<tool_response name="control_lights">{"status":"error","message":"Kitchen lights not responding"}</tool_response>' if include_failure else '<tool_response name="control_lights">{"status":"success","message":"Lights turned on"}</tool_response>'
        
    elif scenario == "weather":
        # Weather scenario
        initial_prompt = base_prompt + """<|im_start|>user
What's the weather like in Amsterdam?
<|im_end|>
<|im_start|>assistant
"""
        tool_response = '<tool_response name="get_weather">{"temperature":15,"conditions":"cloudy"}</tool_response>'
    
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
    
    if scenario == "weather":
        # Add temperature adjustment step for weather scenario
        input("\nPress Enter to continue with temperature adjustment...")
        
        extended_prompt += response2 + """
<|im_end|>
<|im_start|>user
Yes, please set it to 20 degrees
<|im_end|>
<|im_start|>assistant
"""
        
        print("\nStep 3: User requests temperature adjustment")
        print("-" * 40)
        print("\n👤 User:")
        print("Yes, please set it to 20 degrees")
        
        print("\n🤖 Assistant (Generating response...):")
        response3 = generate(model, tokenizer, prompt=extended_prompt, verbose=True)
        print(f"\nResponse: {response3}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Run model inference tests')
    parser.add_argument('--multi-step-test', action='store_true', help='Run multi-step conversation tests')
    parser.add_argument('--interactive', action='store_true', help='Run interactive multi-step test')
    parser.add_argument('--lights', action='store_true', help='Run lights control scenario')
    parser.add_argument('--weather', action='store_true', help='Run weather check scenario')
    parser.add_argument('--failure', action='store_true', help='Include failure responses in scenarios')
    args = parser.parse_args()

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
    checkpoint_path = Path("lora_checkpoints/checkpoint-500.npz")
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

    # Run tests based on command line arguments
    if args.lights:
        run_interactive_multi_step_test(model, tokenizer, "lights", args.failure)
    if args.weather:
        run_interactive_multi_step_test(model, tokenizer, "weather", args.failure)
    if not (args.lights or args.weather):  # Default behavior
        if args.interactive:
            run_interactive_multi_step_test(model, tokenizer, "lights", args.failure)
        elif args.multi_step_test:
            run_multi_step_tests(model, tokenizer)
        else:
            run_single_step_tests(model, tokenizer)

if __name__ == "__main__":
    main() 