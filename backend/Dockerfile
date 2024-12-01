# Use a slim Python image for a smaller footprint
FROM python:3.9-slim

# Set the working directory
WORKDIR /app/backend

# Install system dependencies for pandas and GCS
RUN apt-get update && apt-get install -y \
   gcc \
   python3-dev \
   && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first, for better caching
COPY requirements.txt .

# Install Python dependencies including google-cloud-storage
RUN pip install --no-cache-dir -r requirements.txt \
   && pip install --no-cache-dir google-cloud-storage

# Copy the rest of the application code 
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV WORKERS=1

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
