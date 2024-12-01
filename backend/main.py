import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import io
import re
from google.cloud import storage
from datetime import datetime
from models import TradeData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data from different file formats",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://vola-629904468774.us-central1.run.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Cloud Storage client
try:
    storage_client = storage.Client()
    bucket_name = "vola_bucket"
except Exception as e:
    logger.error(f"Failed to initialize GCS client: {str(e)}")
    storage_client = None

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
CHUNK_SIZE = 8192  # 8KB chunks
MAX_ROWS = 1000

# Define the required columns for each format
REQUIRED_COLUMNS_FORMAT_1 = ["Symbol", "Date/Time", "Quantity", "Proceeds", "Comm/Fee"]
REQUIRED_COLUMNS_FORMAT_2 = ["Description", "DateTime", "Quantity", "Proceeds", "IBCommission"]

def parse_symbol(symbol: str) -> pd.Series:
    """Parse symbol string into components with error handling"""
    try:
        parts = str(symbol).split() if pd.notna(symbol) else []
        return pd.Series({
            'Ticker': parts[0] if len(parts) > 0 else None,
            'Expiry': parts[1] if len(parts) > 1 else None,
            'Strike': parts[2] if len(parts) > 2 else None,
            'Instrument': parts[3] if len(parts) > 3 else None
        })
    except Exception as e:
        logger.error(f"Error parsing symbol {symbol}: {str(e)}")
        return pd.Series({'Ticker': None, 'Expiry': None, 'Strike': None, 'Instrument': None})

def parse_datetime(datetime_str: str, format_type: str) -> pd.Series:
    """Parse datetime string with improved error handling"""
    try:
        if pd.isna(datetime_str):
            return pd.Series({'Date': None, 'Time': None})
            
        if format_type == "Format 1":
            date_part, time_part = datetime_str.split(',')
            if not re.match(r'\d{4}-\d{2}-\d{2}', date_part.strip()):
                raise ValueError("Invalid date format")
            return pd.Series({
                'Date': date_part.strip().replace('-', ''),
                'Time': time_part.strip().replace(':', '')
            })
        else:  # Format 2
            date_part, time_part = datetime_str.split(';')
            return pd.Series({
                'Date': date_part.strip(),
                'Time': time_part.strip()
            })
    except Exception as e:
        logger.error(f"Error parsing datetime {datetime_str}: {str(e)}")
        return pd.Series({'Date': None, 'Time': None})

def convert_to_numeric(value) -> float:
    """Convert string to numeric value with enhanced error handling"""
    try:
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            clean_value = re.sub(r'[^\d.-]', '', value)
            return round(float(clean_value), 4)
        return round(float(value), 4)
    except (ValueError, TypeError) as e:
        logger.warning(f"Error converting value {value}: {str(e)}")
        return 0.0

def find_header_row_and_format(df: pd.DataFrame) -> tuple:
    """Find header row and format type with improved validation"""
    try:
        # Check Format 2 first (headers in first row)
        if all(col in df.columns for col in REQUIRED_COLUMNS_FORMAT_2):
            logger.info("Found Format 2 headers in column names")
            return -1, "Format 2"
        
        # Look for Format 1 headers in the data
        for index, row in df.head(10).iterrows():  # Only check first 10 rows
            cleaned_row = [re.sub(r'[^a-zA-Z]', '', str(cell)).strip().lower() for cell in row]
            if all(re.sub(r'[^a-zA-Z]', '', col).lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_1):
                logger.info(f"Found Format 1 headers at row {index}")
                return index, "Format 1"
        
        return None, None
    except Exception as e:
        logger.error(f"Error finding header row: {str(e)}")
        return None, None

def extract_data(df: pd.DataFrame, header_row: int, format_type: str) -> pd.DataFrame:
    """Extract and process data with memory optimization"""
    try:
        if format_type == "Format 1":
            if header_row >= len(df):
                raise ValueError("Header row index out of bounds")
            
            header = df.iloc[header_row]
            if header.isnull().any():
                raise ValueError("Header contains null values")
                
            df.columns = header
            df = df.iloc[header_row + 1:header_row + MAX_ROWS + 1].copy()
            
            columns_map = {
                "Date/Time": "DateTime",
                "Comm/Fee": "Comm_fee",
                "Symbol": "Symbol",
                "Quantity": "Quantity",
                "Proceeds": "Proceeds"
            }
        else:  # Format 2
            columns_map = {
                "Description": "Symbol",
                "DateTime": "DateTime",
                "Quantity": "Quantity",
                "Proceeds": "Proceeds",
                "IBCommission": "Comm_fee"
            }
            df = df.head(MAX_ROWS).copy()

        # Match columns
        matched_columns = {}
        for original_col, standard_col in columns_map.items():
            if format_type == "Format 2":
                if original_col in df.columns:
                    matched_columns[standard_col] = original_col
            else:
                matched_col = next((col for col in df.columns 
                                  if re.sub(r'[^a-zA-Z]', '', col).strip().lower() == 
                                  re.sub(r'[^a-zA-Z]', '', original_col).strip().lower()), None)
                if matched_col:
                    matched_columns[standard_col] = matched_col

        if len(matched_columns) < len(columns_map):
            raise ValueError("Could not find all required columns")

        # Process data in smaller chunks
        extracted_data = df[list(matched_columns.values())].rename(columns={v: k for k, v in matched_columns.items()})
        extracted_data = extracted_data[extracted_data['DateTime'].notna()]

        # Process symbol and datetime
        symbol_components = extracted_data['Symbol'].apply(parse_symbol)
        datetime_components = extracted_data['DateTime'].apply(lambda x: parse_datetime(x, format_type))
        
        # Concatenate results
        extracted_data = pd.concat([extracted_data, symbol_components, datetime_components], axis=1)

        # Process numeric values
        extracted_data['Proceeds'] = extracted_data['Proceeds'].apply(convert_to_numeric)
        extracted_data['Comm_fee'] = extracted_data['Comm_fee'].apply(convert_to_numeric)
        extracted_data['Net_proceeds'] = extracted_data['Proceeds'] + extracted_data['Comm_fee']

        # Final processing
        final_columns = [
            'Date', 'Time', 'Ticker', 'Expiry', 'Strike', 'Instrument', 
            'Quantity', 'Net_proceeds', 'Symbol', 'DateTime', 'Proceeds', 
            'Comm_fee'
        ]
        
        return extracted_data[final_columns].sort_values(['Date', 'Time'])

    except Exception as e:
        logger.error(f"Error extracting data: {str(e)}")
        raise ValueError(f"Error processing data: {str(e)}")

def save_to_gcs(df: pd.DataFrame, original_filename: str) -> str:
    """Save DataFrame to Google Cloud Storage with error handling"""
    try:
        if storage_client is None:
            raise ValueError("GCS client not initialized")

        columns_to_save = [
            'Date', 'Time', 'Ticker', 'Expiry', 'Strike', 'Instrument', 
            'Quantity', 'Net_proceeds'
        ]
        
        if not all(col in df.columns for col in columns_to_save):
            raise ValueError("Missing required columns for saving")
            
        df_to_save = df[columns_to_save]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = original_filename.rsplit('.', 1)[0]
        filename = f"{base_filename}_{timestamp}.csv"
        
        csv_buffer = io.StringIO()
        df_to_save.to_csv(csv_buffer, index=False)
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"processed_trades/{filename}")
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
        
        return f"gs://{bucket_name}/processed_trades/{filename}"
    
    except Exception as e:
        logger.error(f"Error saving to GCS: {str(e)}")
        raise ValueError(f"Error saving to GCS: {str(e)}")

@app.post("/parse-trades/", response_model=List[TradeData])
async def parse_trades(file: UploadFile = File(...)):
    """Parse trading data from uploaded file with memory optimization"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload .xlsx, .xls, or .csv file"
            )

        # Stream file with size limit
        content = io.BytesIO()
        size = 0
        
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
                )
            content.write(chunk)
        
        content.seek(0)
        
        # Read file with row limit
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(content, nrows=MAX_ROWS)
            else:
                df = pd.read_excel(content, nrows=MAX_ROWS)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(e)}"
            )

        logger.info(f"Initial columns: {df.columns.tolist()}")
        
        # Process data
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
        
        # Save to GCS
        try:
            gcs_path = save_to_gcs(extracted_data, file.filename)
            logger.info(f"File saved to: {gcs_path}")
        except ValueError as e:
            logger.error(f"GCS save error: {str(e)}")
            # Continue processing even if GCS save fails
        
        # Clean NaN values
        for col in extracted_data.select_dtypes(include=['object']).columns:
            extracted_data[col] = extracted_data[col].where(pd.notna(extracted_data[col]), None)
        
        # Convert to response format
        trades_data = extracted_data.head(10).to_dict('records')
        return [TradeData(**trade) for trade in trades_data]
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Trade Parser API",
        "version": "1.0.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload and parse trading data file",
            "/health": "GET - Health check",
            "/": "GET - This information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
