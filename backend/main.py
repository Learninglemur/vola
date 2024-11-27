from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import storage
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import io
import re

# Initialize FastAPI app
app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data from different file formats and storing files in Google Cloud Storage",
    version="1.2.0"
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

# Replace with your bucket name
BUCKET_NAME = "vola_bucket"

def upload_to_gcs(file_content, bucket_name: str, destination_blob_name: str) -> str:
    """
    Uploads a file to Google Cloud Storage and returns the file's public URL.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_content)
    blob.make_public()
    return blob.public_url

def save_dataframe_to_csv_and_upload(df: pd.DataFrame, bucket_name: str, filename: str) -> str:
    """
    Saves a DataFrame to a CSV file and uploads it to Google Cloud Storage.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        bucket_name (str): The GCS bucket name.
        filename (str): The name of the file in GCS.

    Returns:
        str: The public URL of the uploaded CSV file.
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    return upload_to_gcs(csv_content, bucket_name, filename)

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
        public_url = upload_to_gcs(content, BUCKET_NAME, f"uploads/{file.filename}")
        return {"message": "File uploaded successfully", "url": public_url}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@app.post("/parse-trades/", response_model=Dict[str, Optional[str]])
async def parse_trades(file: UploadFile = File(...)):
    """
    Parse trading data from uploaded file, upload the original and parsed files to Google Cloud Storage.
    """
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .xlsx, .xls, or .csv file"
        )
    
    try:
        # Read file content
        content = await file.read()
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        
        print("Initial columns in DataFrame:", df.columns.tolist())

        # Save the original file to Google Cloud Storage
        original_file_url = upload_to_gcs(content, BUCKET_NAME, f"uploads/{file.filename}")

        # Process and parse the file
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
        
        print(f"Detected format: {format_type}")
        
        # Replace NaN values with None for string columns and 0 for numeric columns
        for col in extracted_data.select_dtypes(include=['object']).columns:
            extracted_data[col] = extracted_data[col].where(pd.notna(extracted_data[col]), None)
        
        # Save parsed data as a CSV file and upload to GCS
        parsed_file_url = save_dataframe_to_csv_and_upload(
            extracted_data, BUCKET_NAME, f"parsed/parsed_{file.filename}.csv"
        )
        
        return {
            "message": "File processed successfully",
            "original_file_url": original_file_url,
            "parsed_file_url": parsed_file_url
        }
        
    except Exception as e:
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
            "/upload/": "POST - Upload file to Google Cloud Storage",
            "/": "GET - This information"
        }
    }
