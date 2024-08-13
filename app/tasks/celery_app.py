import httpx

from celery import Celery
from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB_CELERY_BROKER, REDIS_DB_CELERY_BACKEND, CELERY_NUM_WORKERS, INFERENCE_MICROSERVICE_URL, CLOUD_INFERENCE_LAYER
from app.inference.tf_model_manager import TFModelManager
from app.api import schemas

# --- TensorFlow Model Manager ---
model_manager = TFModelManager()

# --- Celery Configuration ---
BROKER_REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_CELERY_BROKER}"
BACKEND_REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_CELERY_BACKEND}"
celery_app = Celery(
    "worker",
    broker=BROKER_REDIS_URL,
    backend=BACKEND_REDIS_URL
)

celery_app.conf.update(
    result_expires=3600,
    task_routes={
        'update_tf_model_task': {'queue': f'model_queue_{k+1}' for k in range(CELERY_NUM_WORKERS)},
        'compute_prediction_task': {'queue': 'prediction_queue'},
    }
)


# --- Tasks ---
@celery_app.task(ignore_result=True)
def update_tf_model_task(tf_model_b64, tf_model_bytesize):
    model_manager.update_model(tf_model_b64, tf_model_bytesize)

@celery_app.task()
def compute_prediction_task(request):
    sensor_data_export = schemas.SensorDataExport(**request)
    metadata: schemas.Metadata = sensor_data_export.metadata
    sensor_data: schemas.SensorData = sensor_data_export.export_value
    sensor_reading: schemas.SensorReading = sensor_data.reading
    
    # --- Prediction ---
    reading_uuid = sensor_reading.uuid
    input_data = sensor_reading.values
    prediction = model_manager.predict(input_data)
    
    # --- Return PredictionResult ---
    prediction_result = schemas.PredictionResult(
        gateway_name=metadata.gateway_name,
        sensor_name=metadata.sensor_name,
        reading_uuid=reading_uuid,
        low_battery=sensor_data.low_battery,
        prediction=prediction
    )
    return prediction_result.model_dump()
    
