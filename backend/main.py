from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage  # Import Google Cloud Storage
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import io
import re

app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data from different file formats and uploading results to Google Cloud Storage",
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

# Replace with your bucket name
BUCKET_NAME = "vola_bucket"

# Google Cloud Storage client
storage_client = storage.Client()

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

# Utility function to upload CSV to Google Cloud Storage
def upload_to_gcs(bucket_name: str, filename: str, content: bytes):
    """
    Uploads a file to Google Cloud Storage and returns the public URL.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(content, content_type="text/csv")
    blob.make_public()
    return blob.public_url

@app.post("/parse-trades/", response_model=List[TradeData])
async def parse_trades(file: UploadFile = File(...)):
    """
    Parse trading data from uploaded file and upload parsed data as CSV to Google Cloud Storage.
    """
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .xlsx, .xls, or .csv file"
        )
    
    try:
        # Read the uploaded file content
        content = await file.read()
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        
        print("Initial columns in DataFrame:", df.columns.tolist())

        # Detect header row and format
        header_row, format_type = find_header_row_and_format(df)
        if format_type is None:
            raise HTTPException(
                status_code=400,
                detail="File format not recognized"
            )
        
        # Extract the parsed data
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

        # Convert extracted data to CSV
        csv_filename = f"parsed_{file.filename}.csv"
        csv_content = extracted_data.to_csv(index=False)

        # Upload the CSV to Google Cloud Storage
        public_url = upload_to_gcs(BUCKET_NAME, csv_filename, csv_content)

        # Convert DataFrame to list of TradeData Pydantic models
        trades_data = extracted_data.to_dict('records')
        return {
            "message": "File processed and uploaded successfully",
            "csv_url": public_url,
            "data": [TradeData(**trade) for trade in trades_data]
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
        "version": "1.1.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload and parse trading data file",
            "/": "GET - This information"
        }
    }
