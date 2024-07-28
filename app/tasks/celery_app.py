import httpx

from celery import Celery
from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB_CELERY, CELERY_NUM_WORKERS, INFERENCE_MICROSERVICE_URL, CLOUD_INFERENCE_LAYER
from app.inference.tf_model_manager import TFModelManager
from app.api import schemas

# --- TensorFlow Model Manager ---
model_manager = TFModelManager()

# --- Celery Configuration ---
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_CELERY}"
celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL
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

@celery_app.task(ignore_result=True)
def compute_prediction_task(request):
    prediction_request_export = schemas.PredictionRequestExport(**request)
    metadata: schemas.Metadata = prediction_request_export.metadata
    prediction_request: schemas.PredictionRequest = prediction_request_export.export_value
    sensor_reading: schemas.SensorReading = prediction_request.reading
    
    # --- Prediction ---
    reading_uuid = sensor_reading.uuid
    input_data = sensor_reading.values
    prediction = model_manager.predict(input_data)
    
    # --- Send Celery Task Result ---
    prediction_result = schemas.PredictionResult(
        reading_uuid=reading_uuid,
        send_timestamp=prediction_request.inference_descriptor.send_timestamp,
        inference_layer=schemas.InferenceLayer(CLOUD_INFERENCE_LAYER),
        prediction=prediction
    )
    celery_task_prediction_result = schemas.CeleryTaskResult(
        metadata=metadata,
        request=prediction_request,
        result=prediction_result
    )

    httpx.put(
        url=f"{INFERENCE_MICROSERVICE_URL}/model/prediction/result",
        json=celery_task_prediction_result.model_dump()
    )
