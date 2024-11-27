FROM python:3.9-slim
# Set working directory to backend folder
WORKDIR /app/backend
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
# Copy backend requirements and install
COPY backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy backend directory contents
COPY backend/
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
