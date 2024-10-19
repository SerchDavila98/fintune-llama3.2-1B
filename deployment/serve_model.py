from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import os
from typing import Optional

class ModelServer:
    """
    A server to handle model loading and prediction.
    """
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, model_path: str):
        """
        Initializes the ModelServer by loading the model and tokenizer.
        
        Args:
            model_path (str): Path to the fine-tuned model directory.
        
        Raises:
            FileNotFoundError: If the model directory does not exist.
            Exception: If loading the model or tokenizer fails.
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model path {model_path} does not exist.")
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        
        try:
            # Load tokenizer from cache or load and cache it
            if model_path in self._tokenizer_cache:
                self.tokenizer = self._tokenizer_cache[model_path]
                self.logger.info(f"Loaded tokenizer from cache for {model_path}.")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._tokenizer_cache[model_path] = self.tokenizer
                self.logger.info(f"Loaded and cached tokenizer for {model_path}.")
            
            # Load model from cache or load and cache it
            if model_path in self._model_cache:
                self.model = self._model_cache[model_path]
                self.logger.info(f"Loaded model from cache for {model_path}.")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.model.eval()
                if torch.cuda.is_available():
                    self.model.to('cuda')
                self._model_cache[model_path] = self.model
                self.logger.info(f"Loaded and cached model for {model_path}.")
        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise e
    
    def predict(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1) -> str:
        """
        Generates a prediction based on the input prompt.
        
        Args:
            prompt (str): The input text prompt.
            max_length (int): The maximum length of the generated sequence.
            num_return_sequences (int): Number of sequences to generate.
        
        Returns:
            str: The generated prediction.
        
        Raises:
            Exception: If prediction fails.
        """
        self.logger.info(f"Received prompt: {prompt}")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.info(f"Generated prediction: {prediction}")
            return prediction
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise e
