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
        "http://localhost:5173",
        "https://vola-629904468774.us-central1.run.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic test endpoint
@app.get("/basic-test/")
async def basic_test():
    """
    Basic test endpoint to verify API is working
    """
    return {"message": "API is working"}

# GCS test endpoint
@app.get("/gcs-test/")
async def gcs_test():
    """
    Test GCS connectivity
    """
    try:
        # Initialize storage client
        storage_client = storage.Client()
        
        # Get project info
        project = storage_client.project
        
        # Try to list buckets
        buckets = list(storage_client.list_buckets())
        bucket_names = [bucket.name for bucket in buckets]
        
        return {
            "status": "success",
            "project": project,
            "buckets": bucket_names
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": str(type(e))
        }

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Trade Parser API",
        "version": "1.1.0",
        "endpoints": {
            "/": "GET - This information",
            "/basic-test/": "GET - Basic API test",
            "/gcs-test/": "GET - Test GCS connectivity"
        }
    }
