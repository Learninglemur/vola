# gunicorn_config.py
import os

# Worker configuration
workers = int(os.getenv('WORKERS', 1))
threads = int(os.getenv('THREADS', 2))
timeout = int(os.getenv('TIMEOUT', 0))

# Binding
bind = f"0.0.0.0:{os.getenv('PORT', 8080)}"

# Worker class
worker_class = 'uvicorn.workers.UvicornWorker'
