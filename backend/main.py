from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import io
import re

app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data from multiple file formats",
    version="1.0.0"
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

# [Previous code for constants and TradeData model remains the same]
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

# [Previous helper functions remain the same]
def parse_symbol(symbol):
    """Parse symbol string into components: Ticker, Expiry, Strike, Instrument"""
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

# [Other helper functions remain the same - parse_datetime, convert_to_numeric, detect_columns_in_row, find_header_row_and_format, extract_data]

async def process_single_file(file: UploadFile) -> pd.DataFrame:
    """
    Process a single file and return extracted data as DataFrame
    """
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for {file.filename}. Please upload .xlsx, .xls, or .csv files"
        )
    
    content = await file.read()
    if file.filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))
    
    header_row, format_type = find_header_row_and_format(df)
    if format_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"File format not recognized for {file.filename}"
        )
    
    extracted_data = extract_data(df, header_row, format_type)
    if extracted_data.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No valid data could be extracted from {file.filename}"
        )
    
    return extracted_data

@app.post("/parse-trades/", response_model=List[TradeData])
async def parse_trades(files: List[UploadFile] = File(...)):
    """
    Parse trading data from multiple uploaded files
    """
    try:
        # Process all files and collect their DataFrames
        all_data = []
        for file in files:
            extracted_df = await process_single_file(file)
            all_data.append(extracted_df)
        
        # Combine all DataFrames
        if not all_data:
            raise HTTPException(
                status_code=400,
                detail="No valid data found in any of the uploaded files"
            )
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort the combined data by Date and Time
        combined_df = combined_df.sort_values(['Date', 'Time'])
        
        # Replace NaN values with None for string columns and 0 for numeric columns
        for col in combined_df.select_dtypes(include=['object']).columns:
            combined_df[col] = combined_df[col].where(pd.notna(combined_df[col]), None)
        
        # Convert to records and create Pydantic models
        trades_data = combined_df.to_dict('records')
        return [TradeData(**trade) for trade in trades_data]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing files: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Trade Parser API",
        "version": "1.0.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload and parse multiple trading data files",
            "/": "GET - This information"
        }
    }
