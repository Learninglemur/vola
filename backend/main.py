from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage  # Google Cloud Storage client
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io
import re

# Initialize FastAPI app
app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data and uploading files to Google Cloud Storage",
    version="1.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "https://vola-629904468774.us-central1.run.app",  # Cloud Run URL
        "*"  # Allow all origins (if needed)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Google Cloud Storage Bucket
BUCKET_NAME = "vola_bucket"  # Replace with your bucket name

# Google Cloud Storage client
storage_client = storage.Client()

def upload_to_gcs(file_content: bytes, destination_blob_name: str) -> str:
    """
    Uploads a file to Google Cloud Storage and returns the file's public URL.
    """
    print(f"Uploading to GCS: Bucket={BUCKET_NAME}, File={destination_blob_name}")
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(file_content)  # Use upload_from_string for raw content
        blob.make_public()
        print(f"File uploaded successfully: {blob.public_url}")
        return blob.public_url
    except Exception as e:
        print(f"Error during upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file to GCS: {str(e)}"
        )

@app.post("/parse-trades/", response_model=dict)
async def parse_trades(file: UploadFile = File(...)):
    """
    Parse trading data from uploaded file and upload the original and parsed files to GCS.
    """
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .xlsx, .xls, or .csv file"
        )
    
    try:
        # Read the file content
        content = await file.read()
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        
        print("Initial columns in DataFrame:", df.columns.tolist())

        # Save the original file to GCS
        public_url = upload_to_gcs(content, file.filename)
        print(f"Original file uploaded: {public_url}")

        # Example data processing (you can add your parsing logic here)
        df['Processed'] = 'Yes'  # Example: Add a dummy column for testing

        # Convert the processed DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        processed_file_content = output.getvalue().encode("utf-8")
        processed_filename = f"processed_{file.filename}"

        # Save the processed file to GCS
        processed_url = upload_to_gcs(processed_file_content, processed_filename)
        print(f"Processed file uploaded: {processed_url}")

        # Return URLs of both original and processed files
        return {
            "message": "File processed and uploaded successfully",
            "original_file_url": public_url,
            "processed_file_url": processed_url
        }
        
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Trade Parser API",
        "version": "1.2.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload and parse trading data file",
            "/": "GET - This information"
        }
    }
