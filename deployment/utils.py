import os
import logging

def get_latest_model_path(model_dir: str) -> str:
    """
    Retrieves the path to the latest fine-tuned model based on modification time.
    
    Args:
        model_dir (str): Directory containing fine-tuned models.
    
    Returns:
        str: Path to the latest model directory.
    
    Raises:
        FileNotFoundError: If no models are found in the directory.
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} does not exist.")
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not subdirs:
        logger.error(f"No fine-tuned models found in {model_dir}.")
        raise FileNotFoundError(f"No fine-tuned models found in {model_dir}.")
    
    latest_dir = max(subdirs, key=os.path.getmtime)
    logger.info(f"Latest model directory: {latest_dir}")
    return latest_dir
