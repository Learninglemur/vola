# gunicorn_config.py
import os

workers = int(os.getenv('WORKERS', 1))
threads = int(os.getenv('THREADS', 2))
timeout = int(os.getenv('TIMEOUT', 0))
bind = f"0.0.0.0:{os.getenv('PORT', 8080)}"
worker_class = "uvicorn.workers.UvicornWorker"
