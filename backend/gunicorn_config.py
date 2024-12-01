# gunicorn_config.py
import os
import multiprocessing

# Worker Configuration
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count()))
worker_class = "uvicorn.workers.UvicornWorker"  # Critical for FastAPI
threads = int(os.getenv('THREADS', 2))
timeout = int(os.getenv('TIMEOUT', 120))  # Increased timeout

# Binding
bind = f"0.0.0.0:{os.getenv('PORT', 8080)}"

# Logging
loglevel = 'info'
accesslog = '-'
errorlog = '-'

# Worker process naming
proc_name = 'fastapi-app'

# Maximum requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50
