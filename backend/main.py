from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io
import re

# Initialize FastAPI app
app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data from different file formats",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "https://vola-629904468774.us-central1.run.app",  # Cloud Run URL
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define required columns for formats
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

def parse_symbol(symbol: str) -> pd.Series:
    """
    Parse symbol string into components: Ticker, Expiry, Strike, Instrument.
    """
    parts = str(symbol).split()
    return pd.Series({
        'Ticker': parts[0] if len(parts) > 0 else None,
        'Expiry': parts[1] if len(parts) > 1 else None,
        'Strike': parts[2] if len(parts) > 2 else None,
        'Instrument': parts[3] if len(parts) > 3 else None
    })

def parse_datetime(datetime_str: str, format_type: str) -> pd.Series:
    """
    Parse datetime string based on format type and return date and time separately.
    """
    if pd.isna(datetime_str):
        return pd.Series({'Date': None, 'Time': None})

    try:
        if format_type == "Format 1":
            date, time = datetime_str.split(",")
            return pd.Series({'Date': date.strip(), 'Time': time.strip()})
        elif format_type == "Format 2":
            date, time = datetime_str.split(";")
            return pd.Series({'Date': date.strip(), 'Time': time.strip()})
    except Exception:
        return pd.Series({'Date': None, 'Time': None})

def convert_to_numeric(value: str) -> float:
    """
    Convert string to numeric value, handling special characters.
    """
    try:
        return round(float(re.sub(r"[^\d.-]", "", value)), 4)
    except (ValueError, TypeError):
        return 0.0

def detect_columns_in_row(row: pd.Series) -> Optional[str]:
    """
    Detect if a row matches the required columns for either format.
    """
    cleaned_row = [re.sub(r"[^a-zA-Z]", "", str(cell)).strip().lower() for cell in row]
    if all(re.sub(r"[^a-zA-Z]", "", col).lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_1):
        return "Format 1"
    elif all(re.sub(r"[^a-zA-Z]", "", col).lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_2):
        return "Format 2"
    return None

def find_header_row_and_format(df: pd.DataFrame) -> (int, Optional[str]):
    """
    Scan rows to find the header row and format type.
    """
    for index, row in df.iterrows():
        format_type = detect_columns_in_row(row)
        if format_type:
            return index, format_type
    return None, None

def extract_data(df: pd.DataFrame, header_row: int, format_type: str) -> pd.DataFrame:
    """
    Extract relevant columns starting from the identified header row.
    """
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
    elif format_type == "Format 2":
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        columns_map = {
            "Description": "Symbol",
            "DateTime": "DateTime",
            "Quantity": "Quantity",
            "Proceeds": "Proceeds",
            "IBCommission": "Comm_fee"
        }

    df = df.rename(columns=columns_map)
    df["Symbol"] = df["Symbol"].apply(parse_symbol)
    df[["Date", "Time"]] = df["DateTime"].apply(lambda x: parse_datetime(x, format_type))
    df["Net_proceeds"] = df["Proceeds"].apply(convert_to_numeric) + df["Comm_fee"].apply(convert_to_numeric)

    return df.reset_index(drop=True)

@app.post("/parse-trades/", response_model=List[TradeData])
async def parse_trades(file: UploadFile = File(...)):
    """
    Parse trading data from an uploaded file.
    """
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload .xlsx, .xls, or .csv files.")
    
    try:
        content = await file.read()
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        
        header_row, format_type = find_header_row_and_format(df)
        if header_row is None or format_type is None:
            raise HTTPException(status_code=400, detail="File format not recognized.")
        
        parsed_data = extract_data(df, header_row, format_type)
        return parsed_data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Trade Parser API is running",
        "version": "1.0.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload and parse trading data file",
            "/": "GET - This information"
        }
    }
