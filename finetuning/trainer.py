from transformers import Trainer, DataCollatorForLanguageModeling, default_data_collator
from datasets import Dataset
import logging
from typing import Optional

def prepare_trainer(model, tokenizer, training_args, dataset: Dataset, data_collator=None) -> Trainer:
    """
    Prepares the Hugging Face Trainer with the given model, tokenizer, and dataset.
    
    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer corresponding to the model.
        training_args: Training arguments for the Trainer.
        dataset: The dataset for training.
        data_collator: Optional data collator. If None, uses default.
    
    Returns:
        Trainer: Configured Trainer instance.
    """
    logger = logging.getLogger(__name__)
    
    if data_collator is None:
        logger.info("Using default data collator.")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    
    # Define evaluation metrics if any
    compute_metrics = None
    if training_args.evaluation_strategy != "no":
        from transformers import EvalPrediction
        import numpy as np

        def compute_metrics(eval_pred: EvalPrediction):
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # Example metric: average length of predictions
            avg_pred_length = np.mean([len(pred.split()) for pred in decoded_preds])
            return {"avg_pred_length": avg_pred_length}
        
        compute_metrics = compute_metrics
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    return trainer
