FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Cloud Run will use port 8080
EXPOSE 8080

# Environment variable for Python buffering
ENV PYTHONUNBUFFERED True

# Run using uvicorn with host 0.0.0.0 to accept all incoming connections
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]