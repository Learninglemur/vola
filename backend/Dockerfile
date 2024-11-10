FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Cloud Run will use port 8080
EXPOSE 8080

# Adjust the command based on your Python framework
CMD ["python", "main.py"]