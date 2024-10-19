from fastapi import APIRouter, HTTPException
from data_generation.data_generator import generate_synthetic_data
from finetuning.finetune import finetune_model
from deployment.serve_model import ModelServer
from deployment.utils import get_latest_model_path
import yaml
import os

router = APIRouter()

@router.post("/train", summary="Train a fine-tuned model based on a use case")
async def train(use_case: str):
    try:
        # Generate synthetic data
        data = generate_synthetic_data(use_case)
        if not data:
            raise ValueError("No data generated for the given use case.")

        # Fine-tune the model
        config = yaml.safe_load(open('config/config.yaml'))
        output_dir = os.path.join(config['model']['finetuned_model_dir'], f"finetuned_{use_case.replace(' ', '_')}")
        os.makedirs(output_dir, exist_ok=True)

        finetune_model(data, output_dir)

        return {"status": "fine-tuning completed", "model_path": output_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", summary="Get prediction from the latest fine-tuned model")
async def predict(prompt: str):
    try:
        config = yaml.safe_load(open('config/config.yaml'))
        model_path = get_latest_model_path(config['model']['finetuned_model_dir'])
        server = ModelServer(model_path)
        prediction = server.predict(prompt)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
