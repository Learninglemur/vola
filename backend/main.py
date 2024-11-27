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

app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data from different formats and storing files in Google Cloud Storage",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://vola-629904468774.us-central1.run.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Cloud Storage client
storage_client = storage.Client()
BUCKET_NAME = "vola_bucket"

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

# Your existing helper functions
def parse_symbol(symbol):
    """Parse symbol string into components"""
    parts = str(symbol).split()
    ticker = parts[0] if len(parts) > 0 else None
    expiry = parts[1] if len(parts) > 1 else None
    strike = parts[2] if len(parts) > 2 else None
    instrument = parts[3] if len(parts) > 3 else None
    
    return pd.Series({
        'Ticker': ticker,
        'Expiry': expiry,
        'Strike': strike,
        'Instrument': instrument
    })

def parse_datetime(datetime_str, format_type):
    """Parse datetime string based on format type"""
    if pd.isna(datetime_str):
        return pd.Series({'Date': None, 'Time': None})
        
    if format_type == "Format 1":
        try:
            date_part, time_part = datetime_str.split(',')
            date = date_part.strip().replace('-', '')
            time = time_part.strip().replace(':', '')
        except ValueError:
            return pd.Series({'Date': None, 'Time': None})
    else:
        try:
            date_part, time_part = datetime_str.split(';')
            date = date_part.strip()
            time = time_part.strip()
        except ValueError:
            return pd.Series({'Date': None, 'Time': None})
    
    return pd.Series({'Date': date, 'Time': time})

def convert_to_numeric(value):
    """Convert string to numeric value"""
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        clean_value = re.sub(r'[^\d.-]', '', value)
        try:
            return round(float(clean_value), 4)
        except ValueError:
            return 0.0
    return round(float(value), 4)

# Continue from previous part...

def detect_columns_in_row(row):
    """Detect if a row matches required columns"""
    cleaned_row = [re.sub(r'[^a-zA-Z]', '', str(cell)).strip().lower() for cell in row]
    logger.info(f"Checking row: {cleaned_row}")

    if all(re.sub(r'[^a-zA-Z]', '', col).lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_1):
        return "Format 1"
    elif all(re.sub(r'[^a-zA-Z]', '', col).lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_2):
        return "Format 2"
    return None

def find_header_row_and_format(df):
    """Find the header row and format type"""
    initial_headers = df.columns.tolist()
    if all(col in initial_headers for col in REQUIRED_COLUMNS_FORMAT_2):
        logger.info("Header found in column headers with Format 2")
        return -1, "Format 2"
    
    for index, row in df.iterrows():
        format_type = detect_columns_in_row(row)
        if format_type == "Format 1":
            logger.info(f"Header found at row {index} with format {format_type}")
            return index, format_type
    return None, None

def extract_data(df, header_row, format_type):
    """Extract and process trade data"""
    if format_type == "Format 1":
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        columns_map = {
            "Date/Time": "DateTime",
            "Comm/Fee": "Comm_fee",
            "Symbol": "Symbol",
            "Quantity": "Quantity",
            "Proceeds": "Proceeds"
        }
        try:
            first_empty_index = df[df['Symbol'].isna() | (df['Symbol'] == '')].index[0]
            df = df.iloc[:first_empty_index]
        except IndexError:
            pass
    elif format_type == "Format 2":
        columns_map = {
            "Description": "Symbol",
            "DateTime": "DateTime",
            "Quantity": "Quantity",
            "Proceeds": "Proceeds",
            "IBCommission": "Comm_fee"
        }

    # Match columns and extract data
    matched_columns = {}
    for original_col, standard_col in columns_map.items():
        if format_type == "Format 2":
            if original_col in df.columns:
                matched_columns[standard_col] = original_col
        else:
            matched_col = next((actual_col for actual_col in df.columns 
                              if re.sub(r'[^a-zA-Z]', '', actual_col).strip().lower() == 
                              re.sub(r'[^a-zA-Z]', '', original_col).strip().lower()), None)
            if matched_col:
                matched_columns[standard_col] = matched_col

    logger.info(f"Matched columns: {matched_columns}")

    if len(matched_columns) < len(columns_map):
        logger.error("Could not find all required columns")
        return pd.DataFrame()

    extracted_data = df[list(matched_columns.values())].rename(columns={v: k for k, v in matched_columns.items()})
    
    if format_type == "Format 1":
        extracted_data = extracted_data[extracted_data['DateTime'].notna()]

    # Process the data
    symbol_components = extracted_data['Symbol'].apply(parse_symbol)
    extracted_data = pd.concat([extracted_data, symbol_components], axis=1)

    datetime_components = extracted_data['DateTime'].apply(lambda x: parse_datetime(x, format_type))
    extracted_data = pd.concat([extracted_data, datetime_components], axis=1)
    
    extracted_data['Proceeds'] = extracted_data['Proceeds'].apply(convert_to_numeric)
    extracted_data['Comm_fee'] = extracted_data['Comm_fee'].apply(convert_to_numeric)
    extracted_data['Net_proceeds'] = extracted_data['Proceeds'] + extracted_data['Comm_fee']
    
    final_columns = [
        'Date', 'Time', 'Ticker', 'Expiry', 'Strike', 'Instrument', 
        'Quantity', 'Net_proceeds', 'Symbol', 'DateTime', 'Proceeds', 
        'Comm_fee'
    ]
    
    extracted_data = extracted_data.sort_values(['Date', 'Time'])
    extracted_data = extracted_data[final_columns]
    
    return extracted_data.head(10)

def upload_to_gcs(content: bytes, filename: str, content_type: str) -> str:
    """Upload file to Google Cloud Storage"""
    try:
        logger.info(f"[DEBUG] Starting upload process for file: {filename}")
        
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        
        # Upload the file
        blob.upload_from_string(content, content_type=content_type)
        
        # Generate signed URL
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET"
        )
        
        return signed_url
    except Exception as e:
        logger.error(f"[DEBUG] Upload failed: {str(e)}")
        raise

@app.post("/parse-trades/", response_model=List[TradeData])
async def parse_trades(file: UploadFile = File(...)):
    """Parse trading data and upload to GCS"""
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .xlsx, .xls, or .csv file"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Upload original file to GCS
        try:
            content_type = file.content_type or 'application/octet-stream'
            url = upload_to_gcs(content, file.filename, content_type)
            logger.info(f"Original file uploaded to GCS: {url}")
        except Exception as upload_error:
            logger.error(f"Failed to upload original file: {str(upload_error)}")
            url = None

        # Parse the file
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        
        logger.info(f"Initial columns: {df.columns.tolist()}")

        # Process the data
        header_row, format_type = find_header_row_and_format(df)
        if format_type is None:
            raise HTTPException(status_code=400, detail="File format not recognized")
        
        extracted_data = extract_data(df, header_row, format_type)
        if extracted_data.empty:
            raise HTTPException(status_code=400, detail="No valid data could be extracted")
        
        # Upload processed data to GCS
        try:
            processed_filename = f"processed_{file.filename.rsplit('.', 1)[0]}.csv"
            processed_content = extracted_data.to_csv(index=False).encode()
            processed_url = upload_to_gcs(processed_content, processed_filename, 'text/csv')
            logger.info(f"Processed data uploaded to GCS: {processed_url}")
        except Exception as process_error:
            logger.error(f"Failed to upload processed file: {str(process_error)}")
            processed_url = None

        # Clean up data for response
        for col in extracted_data.select_dtypes(include=['object']).columns:
            extracted_data[col] = extracted_data[col].where(pd.notna(extracted_data[col]), None)
        
        trades_data = extracted_data.to_dict('records')
        
        response_data = {
            "message": "File processed successfully",
            "original_file_url": url,
            "processed_file_url": processed_url,
            "data": [TradeData(**trade) for trade in trades_data]
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Trade Parser API",
        "version": "1.1.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload, parse, and store trading data",
            "/": "GET - This information"
        }
    }

