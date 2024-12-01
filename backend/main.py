from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CSV Upload API",
    description="API for uploading CSV files to Google Cloud Storage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GCS client
storage_client = storage.Client()
BUCKET_NAME = "vola_bucket"

@app.post("/upload-csv/", 
    summary="Upload CSV file",
    description="Upload a CSV file to Google Cloud Storage",
    response_description="Returns the GCS path where the file was uploaded")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file to Google Cloud Storage.
    
    Parameters:
    - file: The CSV file to upload
    
    Returns:
    - dict: Contains the GCS path and status message
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are allowed"
            )
        
        logger.info(f"Processing file upload: {file.filename}")
        
        # Read file content
        content = await file.read()
        logger.info(f"File size: {len(content)} bytes")
        
        try:
            # Get bucket
            bucket = storage_client.bucket(BUCKET_NAME)
            
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_filename = f"{file.filename.split('.')[0]}_{timestamp}.csv"
            
            # Create blob and upload
            blob = bucket.blob(f"uploads/{new_filename}")
            blob.upload_from_string(content, content_type='text/csv')
            
            gcs_path = f"gs://{BUCKET_NAME}/uploads/{new_filename}"
            logger.info(f"File uploaded successfully to {gcs_path}")
            
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "file_path": gcs_path
            }
            
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading file to Google Cloud Storage: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file upload: {str(e)}"
        )

@app.get("/")
async def root():
    """
    Root endpoint returning API information
    """
    return {
        "message": "CSV Upload API",
        "version": "1.0.0",
        "endpoints": {
            "/upload-csv/": "POST - Upload CSV file to Google Cloud Storage",
            "/docs": "GET - OpenAPI documentation (Swagger UI)",
            "/": "GET - This information"
        }
    }
