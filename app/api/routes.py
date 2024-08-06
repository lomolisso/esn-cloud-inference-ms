import logging
import httpx
import redis
from fastapi import APIRouter, status
from app.api import schemas
from app.tasks.celery_app import update_tf_model_task, compute_prediction_task
from app.core.config import CELERY_NUM_WORKERS, REDIS_HOST, REDIS_PORT, REDIS_DB_HISTORY, CLOUD_API_URL, CLOUD_INFERENCE_LAYER, ADAPTIVE_INFERENCE
from app import utils

router = APIRouter()
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_HISTORY)

# --- Inference endpoints ---

@router.post("/model/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_model(predictive_model: schemas.CloudModel):
    """
    /model/upload endpoint

    This endpoint is used to upload a new model to the server. The model comes as
    a pydantic model object composed of the base64 encoded model .keras file and the
    model byte size for validation purposes. The model is then broadcasted to all workers
    through a specialized queues "model_queue_{i}" where i is the worker number.
    """


    worker_queues = [f"model_queue_{i+1}" for i in range(CELERY_NUM_WORKERS)]
    for k, queue in enumerate(worker_queues, start=1):
        update_tf_model_task.apply_async(
            kwargs=predictive_model.model_dump(),
            queue=queue
        )
        logger.info(f"Model update task submitted to queue: {queue} (worker {k})")

    return {"message": "Model update broadcasted"}

@router.put("/model/prediction/request", status_code=status.HTTP_202_ACCEPTED)
async def prediction_request(prediction_request: schemas.PredictionRequestExport):
    """
    /model/prediction/request endpoint

    This endpoint is used to submit a prediction request to the server. The request comes as
    a pydantic model object composed of a sensor measurement, the name of the sensor, the name
    of the gateway to which the sensor is connected and a UUID for tracking purposes. The request
    is then submitted to a specialized queue "prediction_queue" for processing.
    """

    compute_prediction_task.apply_async(
        kwargs={"request": prediction_request.model_dump()},
        queue="prediction_queue"
    )
    return {"message": "Prediction request received, task submitted to queue: prediction_queue"}

@router.put("/model/prediction/result", status_code=status.HTTP_200_OK)
async def prediction_result(task_result: schemas.CeleryTaskResult):
    """
    /model/prediction/result endpoint

    This endpoint is used to receive a prediction result from the celery worker. The result comes as
    a pydantic model object composed of the reading UUID, the inference layer, the prediction result.
    Before sending the result to the API, the server adds the result from the cloud adaptive heuristic
    to the model object.
    """

    gateway_name = task_result.metadata.gateway_name
    sensor_name = task_result.metadata.sensor_name

    prediction_result: schemas.PredictionResult = task_result.result
    prediction: int = prediction_result.prediction

    heuristic_result = utils.cloud_adaptive_inference_heuristic(
        redis_client=redis_client,
        gateway_name=gateway_name,
        sensor_name=sensor_name,
        prediction=prediction
    ) if ADAPTIVE_INFERENCE else None

    if heuristic_result is not None and heuristic_result != CLOUD_INFERENCE_LAYER:
        layers = {0: "SENSOR_INFERENCE_LAYER", 1: "GATEWAY_INFERENCE_LAYER", 2: "CLOUD_INFERENCE_LAYER", -1: "ERROR"}
        print(f"{sensor_name} inference layer transitioned to {layers[heuristic_result]}")
        utils.clear_prediction_history(redis_client, gateway_name, sensor_name)
        utils.clear_prediction_counter(redis_client, gateway_name, sensor_name)

    export_prediction_result = schemas.PredictionResultExport(
        metadata=task_result.metadata,
        export_value=schemas.PredictionResult(
            reading_uuid=prediction_result.reading_uuid,
            send_timestamp=prediction_result.send_timestamp,
            inference_layer=prediction_result.inference_layer,
            prediction=prediction,
            heuristic_result=heuristic_result
        )
    )

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{CLOUD_API_URL}/export/prediction-result",
            json=export_prediction_result.model_dump()
        )