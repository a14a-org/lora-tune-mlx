"""Utility functions for generating message variations using OpenAI's API."""

import os
from typing import List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model configuration
DEFAULT_MODEL = "gpt-3.5-turbo-instruct"  # Much cheaper than standard models
MAX_TOKENS = 150  # Reduced from 256 since we're just generating variations

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_message_variations(
    original_message: str,
    num_variations: int = 3,
    style: Optional[str] = None,
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL
) -> List[str]:
    """Generate variations of a user message using OpenAI's API.
    
    Args:
        original_message: The original user message to generate variations for
        num_variations: Number of variations to generate
        style: Optional style guidance (e.g. "formal", "casual", "technical")
        temperature: Controls randomness in generation (0.0-1.0)
        model: OpenAI model to use (defaults to cost-effective gpt-3.5-turbo-instruct)
        
    Returns:
        List of generated message variations
    """
    # Construct a concise prompt
    style_guidance = f"in {style} style " if style else ""
    prompt = f"Original: {original_message}\n\nWrite {num_variations} different ways to say this {style_guidance}keeping the exact same meaning. Number each variation:\n"

    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            n=1,
            stop=None,
            frequency_penalty=0.5,  # Encourage variation in wording
            presence_penalty=0.3    # Encourage unique phrasings
        )
        
        # Extract variations from response
        variations_text = response.choices[0].text.strip()
        variations = [
            line.strip().strip('123456789.)"\'').strip()
            for line in variations_text.split('\n')
            if line.strip() and not line.strip().isspace() and len(line.strip()) > 10
        ]
        
        # Ensure we return exactly the requested number of variations
        if len(variations) < num_variations:
            variations.extend([original_message] * (num_variations - len(variations)))
        return variations[:num_variations]
    
    except Exception as e:
        print(f"Error generating variations: {e}")
        # Return original message if API call fails
        return [original_message] * num_variations 