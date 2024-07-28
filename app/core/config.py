import os
import json
from dotenv import load_dotenv

# Retrieve enviroment variables from .env file
load_dotenv()

SECRET_KEY: str = os.environ.get("SECRET_KEY")
INFERENCE_MICROSERVICE_HOST: str = os.environ.get("INFERENCE_MICROSERVICE_HOST")
INFERENCE_MICROSERVICE_PORT: int = int(os.environ.get("INFERENCE_MICROSERVICE_PORT"))
INFERENCE_MICROSERVICE_URL: str = f"http://{INFERENCE_MICROSERVICE_HOST}:{INFERENCE_MICROSERVICE_PORT}/api/v1"

CLOUD_INFERENCE_LAYER = 2
GATEWAY_INFERENCE_LAYER = 1
SENSOR_INFERENCE_LAYER = 0
HEURISTIC_ERROR_CODE = -1

MAX_PREDICTION_HISTORY_LENGTH: int = int(os.environ.get("MAX_PREDICTION_HISTORY_LENGTH", 10))
ABNORMAL_PREDICTION_THRESHOLD: int = min(int(os.environ.get("ABNORMAL_PREDICTION_THRESHOLD", 5)), MAX_PREDICTION_HISTORY_LENGTH)
ABNORMAL_LABELS = json.loads(os.environ.get("ABNORMAL_LABELS", "[2, 3]"))

REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB_CELERY: int = int(os.environ.get("REDIS_DB_CELERY", 0))
REDIS_DB_HISTORY: int = int(os.environ.get("REDIS_DB_HISTORY", 1))

CELERY_NUM_WORKERS: int = int(os.environ.get("CELERY_NUM_WORKERS", 1))
CELERY_CONCURRENCY_LEVEL: int = int(os.environ.get("CELERY_CONCURRENCY_LEVEL", 1))

CLOUD_API_URL: str = os.environ.get("CLOUD_API_URL")

ORIGINS: list = [
    "*"
]
