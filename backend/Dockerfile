# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app/backend

# Install system dependencies for pandas and GCS
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir google-cloud-storage

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV WORKERS=1

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Command to run the application using gunicorn
CMD ["gunicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout", "0", "--worker-class", "uvicorn.workers.UvicornWorker"]
