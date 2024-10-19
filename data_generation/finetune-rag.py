import os
import io
import json
import logging
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from data_generation.data_generator import generate_synthetic_data

router = APIRouter()
logger = logging.getLogger(__name__)

class DatasetResponse(BaseModel):
    status: str
    dataset: List[dict]

@router.post(
    "/upload-pdfs",
    response_model=DatasetResponse,
    summary="Upload PDFs and generate dataset for fine-tuning",
    description="""
    Upload one or more PDF documents. The system will transcribe the content of these PDFs and generate a structured dataset
    based on the provided use case. This dataset can then be used to fine-tune the LLaMA 3.1 405B model.
    """
)
async def upload_pdfs(
    use_case: str = Form(..., description="The specific use case for which the dataset is being generated."),
    files: List[UploadFile] = File(..., description="One or more PDF files to be uploaded and processed.")
) -> DatasetResponse:
    """
    Endpoint to upload PDF files, transcribe them, and generate a dataset for fine-tuning.

    Args:
        use_case (str): The specific use case for which the dataset is being generated.
        files (List[UploadFile]): A list of uploaded PDF files.

    Returns:
        DatasetResponse: Contains the status and the generated dataset.
    """
    logger.info(f"Received upload request for use case: '{use_case}' with {len(files)} files.")

    if not files:
        logger.error("No files uploaded.")
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Extract text from each PDF
    extracted_texts = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Skipping non-PDF file: {file.filename}")
            continue
        try:
            contents = await file.read()
            pdf_reader = PdfReader(io.BytesIO(contents))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if not text.strip():
                logger.warning(f"No text found in PDF: {file.filename}")
            extracted_texts.append(text)
            logger.info(f"Extracted text from '{file.filename}'")
        except Exception as e:
            logger.error(f"Failed to extract text from '{file.filename}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to extract text from '{file.filename}': {str(e)}")

    if not extracted_texts:
        logger.error("No valid text extracted from uploaded PDFs.")
        raise HTTPException(status_code=400, detail="No valid text extracted from uploaded PDFs.")

    # Combine all extracted text
    combined_text = "\n".join(extracted_texts)
    logger.info("Combined text from all PDFs.")

    few_shot_examples = [
        {
            "input": "What is the main topic of the uploaded documents?",
            "output": f"The uploaded documents discuss the following topics:\n{combined_text[:500]}..."  # Truncated for brevity
        },
        {
            "input": "Provide a summary of the key points from the uploaded documents.",
            "output": f"Based on the uploaded documents, here are the key points:\n{combined_text[:500]}..."  # Truncated for brevity
        }
    ]

    try:
        # Generate synthetic dataset using the combined text and use case
        dataset = generate_synthetic_data(
            use_case=use_case,
            num_samples=100,  # Adjust the number of samples as needed
            few_shot_examples=few_shot_examples  # Optional: provide guidance to the AI
        )
        logger.info(f"Generated dataset with {len(dataset)} samples for use case: '{use_case}'")
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dataset: {str(e)}")

    return DatasetResponse(status="Dataset generated successfully.", dataset=dataset)
