from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trade Parser API",
    description="API for parsing trading data",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://vola-629904468774.us-central1.run.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        parts = str(symbol).split()
        return pd.Series({
            "Ticker": parts[0] if len(parts) > 0 else None,
            "Expiry": parts[1] if len(parts) > 1 else None,
            "Strike": parts[2] if len(parts) > 2 else None,
            "Instrument": parts[3] if len(parts) > 3 else None,
        })
    except Exception as e:
        logger.error(f"Error parsing symbol {symbol}: {str(e)}")
        return pd.Series({"Ticker": None, "Expiry": None, "Strike": None, "Instrument": None})

def parse_datetime(datetime_str: str, format_type: str) -> pd.Series:
    try:
        if pd.isna(datetime_str):
            return pd.Series({"Date": None, "Time": None})
        if format_type == "Format 1":
            date, time = datetime_str.split(", ")
            return pd.Series({"Date": date.strip().replace("-", ""), "Time": time.strip().replace(":", "")})
        else:
            date, time = datetime_str.split("; ")
            return pd.Series({"Date": date.strip(), "Time": time.strip()})
    except Exception:
        return pd.Series({"Date": None, "Time": None})

def convert_to_numeric(value: str) -> float:
    try:
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            clean_value = re.sub(r'[^\d.-]', '', value)
            try:
                return round(float(clean_value), 4)
            except ValueError:
                return 0.0
        return round(float(value), 4)
    except Exception:
        return 0.0

def find_header_row_and_format(df: pd.DataFrame) -> tuple:
    try:
        for index, row in df.iterrows():
            cleaned_row = [re.sub(r"[^\w]", "", str(cell)).lower() for cell in row]
            if all(col.lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_1):
                logger.info(f"Found Format 1 at row {index}")
                return index, "Format 1"
            elif all(col.lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_2):
                logger.info(f"Found Format 2 at row {index}")
                return index, "Format 2"
        return None, None
    except Exception as e:
        logger.error(f"Error finding header row: {str(e)}")
        return None, None

def extract_data(df: pd.DataFrame, header_row: int, format_type: str) -> pd.DataFrame:
    try:
        logger.info(f"Extracting data with format {format_type} from row {header_row}")
        if format_type == "Format 1":
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:]
            columns_map = {
                "Date/Time": "DateTime",
                "Comm/Fee": "Comm_fee",
                "Symbol": "Symbol",
                "Quantity": "Quantity",
                "Proceeds": "Proceeds"
            }
        else:
            columns_map = {
                "Description": "Symbol",
                "DateTime": "DateTime",
                "Quantity": "Quantity",
                "Proceeds": "Proceeds",
                "IBCommission": "Comm_fee"
            }

        df = df.rename(columns=columns_map)
        df = df[columns_map.values()]
        
        logger.info("Processing Symbol column")
        df = pd.concat([df, df["Symbol"].apply(parse_symbol)], axis=1)
        
        logger.info("Processing DateTime column")
        df[["Date", "Time"]] = df["DateTime"].apply(lambda x: parse_datetime(x, format_type))
        
        logger.info("Processing numeric columns")
        df["Proceeds"] = df["Proceeds"].apply(convert_to_numeric)
        df["Comm_fee"] = df["Comm_fee"].apply(convert_to_numeric)
        df["Net_proceeds"] = df["Proceeds"] + df["Comm_fee"]
        
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error extracting data: {str(e)}")
        raise

@app.post("/parse-trades/", response_model=List[TradeData])
async def parse_trades(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload .xlsx, .xls, or .csv files.")
    
    try:
        logger.info(f"Processing file: {file.filename}")
        content = await file.read()
        
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
        
        logger.info(f"File loaded successfully. Shape: {df.shape}")
        
        header_row, format_type = find_header_row_and_format(df)
        if header_row is None or format_type is None:
            raise HTTPException(status_code=400, detail="File format not recognized.")
        
        parsed_data = extract_data(df, header_row, format_type)
        if parsed_data.empty:
            raise HTTPException(status_code=400, detail="No valid data could be extracted from the file")
            
        result = {"message": "File processed successfully", "data": parsed_data.to_dict(orient="records")}
        logger.info("File processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Trade Parser API is running",
        "version": "2.0.0",
        "endpoints": {
            "/parse-trades/": "POST - Upload and parse trading data file",
            "/": "GET - This information"
        }
    }
