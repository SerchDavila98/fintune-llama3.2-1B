import os
import yaml
import logging
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from .trainer import prepare_trainer
from .utils import preprocess_data, save_training_metrics
from typing import List, Dict

def finetune_model(raw_data: List[Dict[str, str]], output_dir: str, use_case: str) -> None:
    """
    Fine-tunes the base model using the provided synthetic dataset.
    
    Args:
        raw_data (List[Dict[str, str]]): The synthetic dataset to use for fine-tuning.
        output_dir (str): Directory to save the fine-tuned model.
        use_case (str): The specific use case being fine-tuned for.
    
    Raises:
        Exception: If any step in the fine-tuning process fails.
    """
    # Load configuration
    config = yaml.safe_load(open('config/config.yaml'))
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    try:
        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model: {config['model']['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
        model = AutoModelForCausalLM.from_pretrained(config['model']['base_model'])
        
        # Preprocess data
        logger.info("Preprocessing data...")
        dataset = preprocess_data(raw_data, tokenizer)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            save_steps=config['training']['save_steps'],
            save_total_limit=config['training']['save_total_limit'],
            logging_dir=config['logging']['logging_dir'],
            logging_steps=config['training']['logging_steps'],
            evaluation_strategy=config['training'].get('evaluation_strategy', 'no'),
            load_best_model_at_end=config['training'].get('load_best_model_at_end', False),
            metric_for_best_model=config['training'].get('metric_for_best_model', None),
            greater_is_better=config['training'].get('greater_is_better', None),
            seed=config['training'].get('seed', 42),
        )
        
        # Prepare trainer
        logger.info("Preparing trainer...")
        trainer = prepare_trainer(model, tokenizer, training_args, dataset, config['training'].get('data_collator', None))
        
        # Start training
        logger.info("Starting training...")
        training_output = trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        # Save training metrics
        if 'metrics' in training_output:
            save_training_metrics(training_output.metrics, output_dir)
            logger.info("Training metrics saved.")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise e
