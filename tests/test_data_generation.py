import unittest
from unittest.mock import patch, MagicMock
from data_generation.data_generator import generate_synthetic_data

class TestDataGeneration(unittest.TestCase):
    @patch('data_generation.data_generator.OpenAI')
    def test_generate_synthetic_data_success(self, mock_openai):
        # Mock API response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='[{"input": "Sample question?", "output": "Sample answer."}]'))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        use_case = "customer support"
        data = generate_synthetic_data(use_case, num_samples=10)
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIn("input", data[0])
        self.assertIn("output", data[0])
    
    @patch('data_generation.data_generator.OpenAI')
    def test_generate_synthetic_data_invalid_json(self, mock_openai):
        # Mock API response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='Invalid JSON response'))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        use_case = "customer support"
        with self.assertRaises(ValueError):
            generate_synthetic_data(use_case, num_samples=10)
    
    @patch('data_generation.data_generator.OpenAI')
    def test_generate_synthetic_data_missing_keys(self, mock_openai):
        # Mock API response with missing keys
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='[{"input": "Sample question?"}]'))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        use_case = "customer support"
        with self.assertRaises(ValueError):
            generate_synthetic_data(use_case, num_samples=10)
    
    @patch('data_generation.data_generator.OpenAI')
    def test_generate_synthetic_data_empty_response(self, mock_openai):
        # Mock API response with empty data
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='[]'))
        ]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        use_case = "customer support"
        data = generate_synthetic_data(use_case, num_samples=0)
        self.assertEqual(data, [])
    
    def test_generate_synthetic_data_argument_validation(self):
        # Test with invalid arguments
        from data_generation.data_generator import generate_synthetic_data
        with self.assertRaises(TypeError):
            generate_synthetic_data()  # Missing required 'use_case' argument

if __name__ == '__main__':
    unittest.main()
