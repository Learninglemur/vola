FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variable for Python buffering
ENV PYTHONUNBUFFERED True

# Expose the port
EXPOSE 8080

# Run using gunicorn with uvicorn worker
CMD exec gunicorn --config gunicorn_config.py main:app
