from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="Automated Fine-Tuning Pipeline API",
    description="API for training and deploying fine-tuned LLaMA models",
    version="1.0.0",
)

app.include_router(router)
