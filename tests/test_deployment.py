import unittest
from unittest.mock import patch, MagicMock
from deployment.serve_model import ModelServer
import os

class TestDeployment(unittest.TestCase):
    @patch('deployment.serve_model.AutoTokenizer')
    @patch('deployment.serve_model.AutoModelForCausalLM')
    def test_model_server_success(self, mock_model, mock_tokenizer):
        # Mock tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_model.generate.return_value = MagicMock(input_ids=[[1,2,3]])
        mock_tokenizer.decode.return_value = "Decoded prediction."
        
        model_path = "./models/finetuned_models/test_model"
        os.makedirs(model_path, exist_ok=True)
        
        server = ModelServer(model_path)
        prediction = server.predict("Test prompt")
        
        self.assertEqual(prediction, "Decoded prediction.")
        mock_tokenizer.from_pretrained.assert_called_with(model_path)
        mock_model.from_pretrained.assert_called_with(model_path)
        mock_model.generate.assert_called()
        mock_tokenizer.decode.assert_called()
        
        # Cleanup
        if os.path.exists(model_path):
            os.rmdir(model_path)
    
    def test_model_server_missing_model(self):
        model_path = "./models/finetuned_models/nonexistent_model"
        
        with self.assertRaises(FileNotFoundError):
            server = ModelServer(model_path)
    
    @patch('deployment.serve_model.AutoTokenizer')
    @patch('deployment.serve_model.AutoModelForCausalLM')
    def test_model_server_prediction_error(self, mock_model, mock_tokenizer):
        # Mock tokenizer and model with generate error
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_model.generate.side_effect = Exception("Generation failed")
        
        model_path = "./models/finetuned_models/test_model_error"
        os.makedirs(model_path, exist_ok=True)
        
        server = ModelServer(model_path)
        with self.assertRaises(Exception) as context:
            prediction = server.predict("Test prompt")
        
        self.assertIn("Generation failed", str(context.exception))
        
        # Cleanup
        if os.path.exists(model_path):
            os.rmdir(model_path)

if __name__ == '__main__':
    unittest.main()
