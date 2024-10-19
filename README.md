# Automated Fine-Tuning Pipeline for LLaMA Models

## Overview
This project automates the fine-tuning of LLaMA models using synthetic datasets generated via the AI/ML API. It provides a RESTful API to train and deploy fine-tuned models based on user-defined use cases.

## Features
- **Synthetic Data Generation**: Uses the AI/ML API to generate datasets tailored to specific use cases.
- **Model Fine-Tuning**: Fine-tunes the `meta-llama/Llama-3.2-1B-Instruct` model using Hugging Face's Transformers library.
- **Local Deployment**: Serves the fine-tuned model locally for inference.
- **RESTful API**: Exposes endpoints to trigger training and obtain predictions.

## Setup

### 1. **Clone the repository**
```bash
git clone https://github.com/yourusername/automated_finetuning_pipeline.git
cd automated_finetuning_pipeline
