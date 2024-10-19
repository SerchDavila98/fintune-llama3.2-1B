import unittest
from unittest.mock import patch, MagicMock
from finetuning.finetune import finetune_model
import os
import shutil

class TestFinetuning(unittest.TestCase):
    @patch('finetuning.finetune.AutoTokenizer')
    @patch('finetuning.finetune.AutoModelForCausalLM')
    @patch('finetuning.finetune.Trainer')
    def test_finetune_model_success(self, mock_trainer, mock_model, mock_tokenizer):
        # Mock tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Mock Trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock training
        mock_trainer_instance.train.return_value = MagicMock(metrics={"loss": 0.1})
        
        raw_data = [
            {"input": "How can I reset my password?", "output": "Instructions to reset password."},
            {"input": "What is the refund policy?", "output": "Details about refund policy."},
        ]
        output_dir = "./models/finetuned_models/test_finetune"
        use_case = "customer support"
        
        # Ensure the output directory is clean
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        finetune_model(raw_data, output_dir, use_case)
        
        # Assertions
        mock_tokenizer.from_pretrained.assert_called_with('meta-llama/Llama-3.2-1B-Instruct')
        mock_model.from_pretrained.assert_called_with('meta-llama/Llama-3.2-1B-Instruct')
        mock_trainer.assert_called()
        mock_trainer_instance.train.assert_called()
        mock_trainer_instance.save_model.assert_called_with(output_dir)
        mock_trainer_instance.tokenizer.save_pretrained.assert_called_with(output_dir)
        
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    
    @patch('finetuning.finetune.AutoTokenizer')
    @patch('finetuning.finetune.AutoModelForCausalLM')
    @patch('finetuning.finetune.Trainer')
    def test_finetune_model_failure(self, mock_trainer, mock_model, mock_tokenizer):
        # Mock tokenizer and model
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer load failed")
        mock_model.from_pretrained.return_value = MagicMock()
        
        raw_data = [
            {"input": "Sample input?", "output": "Sample output."},
        ]
        output_dir = "./models/finetuned_models/test_finetune_failure"
        use_case = "customer support"
        
        with self.assertRaises(Exception) as context:
            finetune_model(raw_data, output_dir, use_case)
        
        self.assertIn("Tokenizer load failed", str(context.exception))
        
        # Ensure no model is saved
        self.assertFalse(os.path.exists(output_dir))

if __name__ == '__main__':
    unittest.main()
