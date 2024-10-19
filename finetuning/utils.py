import pandas as pd
from datasets import Dataset
from typing import List, Dict
import logging
import json
import os

def preprocess_data(raw_data: List[Dict[str, str]], tokenizer, max_length: int = 512) -> Dataset:
    """
    Preprocesses raw data by tokenizing and formatting it for training.
    
    Args:
        raw_data (List[Dict[str, str]]): The raw synthetic dataset.
        tokenizer: The tokenizer to use for processing text.
        max_length (int): Maximum sequence length. Defaults to 512.
    
    Returns:
        Dataset: A Hugging Face Dataset object ready for training.
    
    Raises:
        ValueError: If the input data does not conform to expected structure.
    """
    logger = logging.getLogger(__name__)
    
    if not isinstance(raw_data, list):
        logger.error("Raw data is not a list.")
        raise ValueError("Raw data must be a list of dictionaries.")
    
    for item in raw_data:
        if not isinstance(item, dict):
            logger.error("One of the data samples is not a dictionary.")
            raise ValueError("Each data sample must be a dictionary.")
        if "input" not in item or "output" not in item:
            logger.error("Data sample missing 'input' or 'output' keys.")
            raise ValueError("Each data sample must contain 'input' and 'output' keys.")
    
    # Create DataFrame
    df = pd.DataFrame(raw_data)
    logger.info(f"DataFrame created with {len(df)} samples.")
    
    # Tokenize inputs and outputs
    def tokenize_function(examples):
        return tokenizer(examples['input'], truncation=True, padding='max_length', max_length=max_length)
    
    tokenized_inputs = df['input'].tolist()
    tokenized_outputs = df['output'].tolist()
    
    tokenized_data = tokenizer(tokenized_inputs, truncation=True, padding='max_length', max_length=max_length)
    labels = tokenizer(tokenized_outputs, truncation=True, padding='max_length', max_length=max_length)['input_ids']
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels
    ]
    
    tokenized_data['labels'] = labels
    
    dataset = Dataset.from_dict(tokenized_data)
    logger.info("Data has been tokenized and formatted for training.")
    
    return dataset

def save_training_metrics(metrics: Dict, output_dir: str) -> None:
    """
    Saves the training metrics to a JSON file in the output directory.
    
    Args:
        metrics (Dict): The metrics dictionary from training.
        output_dir (str): Directory where to save the metrics.
    
    Raises:
        IOError: If saving the metrics fails.
    """
    logger = logging.getLogger(__name__)
    
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Training metrics saved to {metrics_path}")
    except IOError as e:
        logger.error(f"Failed to save training metrics: {str(e)}")
        raise e
