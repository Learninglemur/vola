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

def parse_symbol(symbol):
    """
    Parse symbol string into components: Ticker, Expiry, Strike, Instrument
    """
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
    """
    Parse datetime string based on format type and return date and time separately
    """
    if pd.isna(datetime_str):
        return pd.Series({'Date': None, 'Time': None})
        
    if format_type == "Format 1":
        try:
            date_part, time_part = datetime_str.split(',')
            date = date_part.strip().replace('-', '')
            time = time_part.strip().replace(':', '')
        except ValueError:
            return pd.Series({'Date': None, 'Time': None})
    else:  # Format 2
        try:
            date_part, time_part = datetime_str.split(';')
            date = date_part.strip()
            time = time_part.strip()
        except ValueError:
            return pd.Series({'Date': None, 'Time': None})
    
    return pd.Series({'Date': date, 'Time': time})

def convert_to_numeric(value):
    """
    Convert string to numeric value, handling different formats
    """
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        clean_value = re.sub(r'[^\d.-]', '', value)
        try:
            return round(float(clean_value), 4)
        except ValueError:
            return 0.0
    return round(float(value), 4)

def detect_columns_in_row(row):
    """
    Detect if a row matches the required columns for either format.
    """
    cleaned_row = [re.sub(r'[^a-zA-Z]', '', str(cell)).strip().lower() for cell in row]
    print(f"Checking row: {cleaned_row}")

    if all(re.sub(r'[^a-zA-Z]', '', col).lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_1):
        return "Format 1"
    
    elif all(re.sub(r'[^a-zA-Z]', '', col).lower() in cleaned_row for col in REQUIRED_COLUMNS_FORMAT_2):
        return "Format 2"
    
    return None

def find_header_row_and_format(df):
    """
    Scan rows progressively to find the header row and format type.
    """
    initial_headers = df.columns.tolist()
    if all(col in initial_headers for col in REQUIRED_COLUMNS_FORMAT_2):
        print(f"Header found in column headers with Format 2")
        return -1, "Format 2"
    
    for index, row in df.iterrows():
        format_type = detect_columns_in_row(row)
        if format_type == "Format 1":
            print(f"Header found at row {index} with format {format_type}")
            return index, format_type
    return None, None

def extract_data(df, header_row, format_type):
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

    # Match and extract columns
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

    print(f"Matched columns: {matched_columns}")

    if len(matched_columns) < len(columns_map):
        print("Error: Could not find all required columns in the header row.")
        return pd.DataFrame()

    extracted_data = df[list(matched_columns.values())].rename(columns={v: k for k, v in matched_columns.items()})

    if format_type == "Format 1":
        extracted_data = extracted_data[extracted_data['DateTime'].notna()]

    # Parse Symbol into separate columns
    symbol_components = extracted_data['Symbol'].apply(parse_symbol)
    extracted_data = pd.concat([extracted_data, symbol_components], axis=1)

    # Parse DateTime into Date and Time
    datetime_components = extracted_data['DateTime'].apply(lambda x: parse_datetime(x, format_type))
    extracted_data = pd.concat([extracted_data, datetime_components], axis=1)
    
    # Convert numeric values and calculate net proceeds
    extracted_data['Proceeds'] = extracted_data['Proceeds'].apply(convert_to_numeric)
    extracted_data['Comm_fee'] = extracted_data['Comm_fee'].apply(convert_to_numeric)
    extracted_data['Net_proceeds'] = extracted_data['Proceeds'] + extracted_data['Comm_fee']
    
    # Reorder columns
    final_columns = [
        'Date', 'Time', 'Ticker', 'Expiry', 'Strike', 'Instrument', 
        'Quantity', 'Net_proceeds', 'Symbol', 'DateTime', 'Proceeds', 
        'Comm_fee'
    ]
    
    # Sort by Date and Time
    extracted_data = extracted_data.sort_values(['Date', 'Time'])
    extracted_data = extracted_data[final_columns]
    
    return extracted_data

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
    
    print(f"Processing file: {file.filename}")
    print("Initial columns in DataFrame:", df.columns.tolist())
    
    header_row, format_type = find_header_row_and_format(df)
    if format_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"File format not recognized for {file.filename}"
        )
    
    print(f"Detected format: {format_type} for file: {file.filename}")
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
