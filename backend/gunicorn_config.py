import os
import multiprocessing

# Basic configuration
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
workers = 1  # Single worker for better memory management
worker_class = 'uvicorn.workers.UvicornWorker'

# Worker configurations
worker_connections = 1000
timeout = 120
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Memory management
worker_tmp_dir = '/dev/shm'
max_requests = 1000
max_requests_jitter = 50

# Process naming
proc_name = 'trade-parser'

def on_starting(server):
    """Log when server is starting"""
    server.log.info("Starting trade parser server")

def worker_abort(worker):
    """Log worker abort"""
    worker.log.info(f"worker {worker.pid} aborted")

def worker_exit(server, worker):
    """Log worker exit"""
    server.log.info(f"worker {worker.pid} exited")
