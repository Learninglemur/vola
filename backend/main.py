from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import storage
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import datetime
import logging
import io
import re
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data from different file formats and storing files in Google Cloud Storage",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "https://vola-629904468774.us-central1.run.app",  # Cloud Run URL
        "*"  # Or this to allow all origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the required columns for each format
REQUIRED_COLUMNS_FORMAT_1 = ["Symbol", "Date/Time", "Quantity", "Proceeds", "Comm/Fee"]
REQUIRED_COLUMNS_FORMAT_2 = ["Description", "DateTime", "Quantity", "Proceeds", "IBCommission"]

class TradeData(BaseModel):
    Date: Optional[str]
    Time: Optional[str]
    Ticker: Optional[str]
    Expiry: Optional[str]
    Strike: Optional[str]
    Instrument: Optional[str]
    Quantity: float
    Net_proceeds: float
    Symbol: Optional[str]
    DateTime: Optional[str]
    Proceeds: float
    Comm_fee: float

# Google Cloud Storage client
storage_client = storage.Client()
BUCKET_NAME = "vola_bucket"

def upload_to_gcs(content: bytes, filename: str, content_type: str) -> str:
    """
    Uploads a file to Google Cloud Storage with detailed error logging.
    """
    try:
        logger.info(f"[DEBUG] Starting upload process for file: {filename}")
        logger.info(f"[DEBUG] Content length: {len(content)} bytes")
        logger.info(f"[DEBUG] Content type: {content_type}")
        
        # Print current credentials
        credentials = storage_client.credentials
        logger.info(f"[DEBUG] Using credentials: {credentials.__class__.__name__}")
        logger.info(f"[DEBUG] Project ID: {storage_client.project}")
        
        # Try to list buckets to verify credentials
        try:
            buckets = list(storage_client.list_buckets(max_results=1))
            logger.info(f"[DEBUG] Successfully listed buckets: {[b.name for b in buckets]}")
        except Exception as e:
            logger.error(f"[DEBUG] Failed to list buckets: {str(e)}")
        
        # Get bucket
        try:
            bucket = storage_client.bucket(BUCKET_NAME)
            logger.info(f"[DEBUG] Got bucket reference: {bucket.name}")
        except Exception as e:
            logger.error(f"[DEBUG] Failed to get bucket: {str(e)}")
            raise
        
        # Create blob
        try:
            blob = bucket.blob(filename)
            logger.info(f"[DEBUG] Created blob reference: {blob.name}")
        except Exception as e:
            logger.error(f"[DEBUG] Failed to create blob: {str(e)}")
            raise
        
        # Upload content
        try:
            blob.upload_from_string(content, content_type=content_type)
            logger.info("[DEBUG] Successfully uploaded content")
        except Exception as e:
            logger.error(f"[DEBUG] Failed to upload content: {str(e)}")
            raise
        
        # Generate signed URL
        try:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(hours=1),
                method="GET"
            )
            logger.info("[DEBUG] Generated signed URL")
            return signed_url
        except Exception as e:
            logger.error(f"[DEBUG] Failed to generate signed URL: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"[DEBUG] Top level error in upload_to_gcs: {str(e)}")
        logger.exception("[DEBUG] Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload to GCS: {str(e)}"
        )

@app.post("/upload/")
async def upload_file_to_gcs(file: UploadFile = File(...)):
    """
    Upload a file to Google Cloud Storage.
    """
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .xlsx, .xls, or .csv file"
        )
    
    try:
        content = await file.read()
        content_type = file.content_type or 'application/octet-stream'
        logger.info(f"[DEBUG] Attempting to upload: {file.filename}, Content-Type: {content_type}")
        url = upload_to_gcs(content, file.filename, content_type)
        return {"message": "File uploaded successfully", "url": url}
    except Exception as e:
        logger.error(f"[DEBUG] Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@app.post("/parse-trades/", response_model=List[TradeData])
async def parse_trades(file: UploadFile = File(...)):
    """
    Parse trading data from uploaded file and upload it to Google Cloud Storage.
    """
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .xlsx, .xls, or .csv file"
        )
    
    try:
        # Read file content
        content = await file.read()
        logger.info(f"[DEBUG] File read successfully: {file.filename}")
        
        # Try GCS upload first
        try:
            content_type = file.content_type or 'application/octet-stream'
            logger.info(f"[DEBUG] Attempting GCS upload with content-type: {content_type}")
            url = upload_to_gcs(content, file.filename, content_type)
            logger.info(f"[DEBUG] GCS upload successful, URL: {url}")
        except Exception as upload_error:
            logger.error(f"[DEBUG] GCS upload failed: {str(upload_error)}")
            # Continue with parsing even if upload fails
        
        # Parse the file
        file_content = io.BytesIO(content)
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_content)
        else:
            df = pd.read_csv(file_content)
        
        logger.info(f"[DEBUG] Initial columns in DataFrame: {df.columns.tolist()}")

        header_row, format_type = find_header_row_and_format(df)
        if format_type is None:
            raise HTTPException(
                status_code=400,
                detail="File format not recognized"
            )
        
        extracted_data = extract_data(df, header_row, format_type)
        if extracted_data.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid data could be extracted from the file"
            )
        
        logger.info(f"[DEBUG] Detected format: {format_type}")
        
        # Replace NaN values with None for string columns and 0 for numeric columns
        for col in extracted_data.select_dtypes(include=['object']).columns:
            extracted_data[col] = extracted_data[col].where(pd.notna(extracted_data[col]), None)
        
        # Convert to records and create Pydantic models
        trades_data = extracted_data.to_dict('records')
        return {
            "message": "File processed successfully",
            "url": url if 'url' in locals() else None,
            "data": [TradeData(**trade) for trade in trades_data]
        }
        
    except Exception as e:
        logger.error(f"[DEBUG] Error processing file: {str(e)}")
        logger.exception("[DEBUG] Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Trade Parser API",
        "version": "1.1.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload and parse trading data file",
            "/upload/": "POST - Upload file to Google Cloud Storage",
            "/": "GET - This information"
        }
    }
