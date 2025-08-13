from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables if you're using a .env file

# It's good practice to use environment variables for broker/backend URLs
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery_app = Celery(
    'pockethunter_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks']  # List of modules to import when a worker starts
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True # Important for robust startup
)

if __name__ == '__main__':
    celery_app.start() 