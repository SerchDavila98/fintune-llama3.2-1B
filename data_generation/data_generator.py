import os
import json
import yaml
from openai import OpenAI
from typing import List, Optional
from .utils import load_config

def generate_synthetic_data(use_case: str, 
                           num_samples: int = 100, 
                           few_shot_examples: Optional[List[dict]] = None) -> List[dict]:
    """
    Generates synthetic data for a given use case using the AI/ML API.

    Args:
        use_case (str): The specific use case for which to generate the dataset.
        num_samples (int): The number of data samples to generate. Default is 100.
        few_shot_examples (Optional[List[dict]]): A list of example data points to guide the AI.
    
    Returns:
        List[dict]: A list of data samples adhering to the defined structure.
    """
    config = load_config('config/config.yaml')
    client = OpenAI(
        api_key=config['api']['aimlapi_key'],
        base_url=config['api']['aimlapi_base_url'],
    )

    # Define the base prompt
    prompt = f"""
    You are an AI assistant specialized in generating high-quality datasets for machine learning tasks.
    
    **Use Case:** {use_case}
    
    **Instructions:**
    - Generate a dataset with {num_samples} samples.
    - Each data point should be a JSON object.
    - Follow the structure defined below.
    - Ensure the data is diverse and covers various aspects of the use case.
    
    **Dataset Structure:**
    ```json
    [
        {
            "input": "<input_text>",
            "output": "<output_text>"
        },
        ...
    ]
    ```
    
    **Two-Shot Examples:**
    ```json
    [
        {
            "input": "How can I reset my password?",
            "output": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions sent to your email."
        },
        {
            "input": "What is the refund policy?",
            "output": "Our refund policy allows you to return products within 30 days of purchase for a full refund, provided the items are in original condition."
        }
    ]
    """
    
    # Append few-shot examples if provided
    if few_shot_examples:
        prompt += "\n\n**Few-Shot Examples:**\n```json\n" + json.dumps(few_shot_examples, indent=4) + "\n```\n"

    # Append the instruction to generate the dataset
    prompt += "\n\n**Generated Dataset:**"

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant specialized in generating high-quality datasets for machine learning tasks.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=3000,  # Adjust based on expected response size
        temperature=0.7,   # Control the randomness of the output
    )

    message = response.choices[0].message.content.strip()

    # Extract JSON data from the response
    try:
        # Attempt to parse the entire message as JSON
        data = json.loads(message)
    except json.JSONDecodeError:
        try:
            # If the response contains the JSON within text, extract it
            json_start = message.index('[')
            json_str = message[json_start:]
            data = json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            # As a last resort, split lines and attempt to parse each line as a JSON object
            data = []
            for line in message.split('\n'):
                line = line.strip().rstrip(',')
                if line.startswith('{') and line.endswith('}'):
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                    except json.JSONDecodeError:
                        continue
            if not data:
                raise ValueError("Failed to parse the synthetic data from the API response.")
    
    # Validate the structure of the data
    if not isinstance(data, list):
        raise ValueError("The generated data is not a list of dictionaries.")
    
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each data sample should be a dictionary.")
        if "input" not in item or "output" not in item:
            raise ValueError("Each data sample must contain 'input' and 'output' keys.")
    
    return data
