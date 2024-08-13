import logging
from fastapi import APIRouter, status
from app.api import schemas
from celery.result import AsyncResult
from app.tasks.celery_app import update_tf_model_task, compute_prediction_task
from app.core.config import CELERY_NUM_WORKERS, CLOUD_INFERENCE_LAYER, ADAPTIVE_INFERENCE
from app import utils

router = APIRouter()
logger = logging.getLogger(__name__)

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
async def prediction_request(prediction_request: schemas.SensorDataExport):
    """
    /model/prediction/request endpoint

    This endpoint is used to submit a prediction request to the server. The request comes as
    a pydantic model object composed of a sensor measurement, the name of the sensor, the name
    of the gateway to which the sensor is connected and a UUID for tracking purposes. The request
    is then submitted to a specialized queue "prediction_queue" for processing.
    """

    task = compute_prediction_task.apply_async(
        kwargs={"request": prediction_request.model_dump()},
        queue="prediction_queue"
    )
    return {
        "message": "Prediction request received, task submitted to queue: prediction_queue",
        "task_id": task.id
    }

@router.get("/model/prediction/result/{task_id}", status_code=status.HTTP_200_OK)
async def get_prediction_result(task_id: str):
    """
    /model/prediction/result/{task_id} endpoint

    This endpoint is used to retrieve the result of a prediction request. The task_id is used
    to query the celery worker for the result of the prediction request. The result is then
    returned to the client as a pydantic model object composed of the reading UUID, the inference
    layer, the prediction result and the heuristic result.
    """

    task_result = AsyncResult(task_id)

    if not task_result.ready():
        return schemas.TaskResult(status="PENDING")
    elif task_result.ready() and task_result.failed():
        return schemas.TaskResult(status="FAILURE")
    else:
        prediction_task_result = schemas.PredictionResult(**task_result.result)
        gateway_name = prediction_task_result.gateway_name
        sensor_name = prediction_task_result.sensor_name

        prediction_result = prediction_task_result.prediction
        heuristic_result = None
        if ADAPTIVE_INFERENCE:
            heuristic_result = utils.cloud_adaptive_inference_heuristic(
                prediction_result=prediction_task_result,
            )
            if heuristic_result is not None and heuristic_result != CLOUD_INFERENCE_LAYER:
                layers = {0: "SENSOR_INFERENCE_LAYER", 1: "GATEWAY_INFERENCE_LAYER", 2: "CLOUD_INFERENCE_LAYER", -1: "ERROR"}
                if heuristic_result is None:
                    print("ERROR: Heuristic returned None")
                
                print(f"{sensor_name} inference layer transitioned to {layers[heuristic_result]}")
                utils.clear_prediction_history(gateway_name, sensor_name)
                utils.clear_prediction_counter(gateway_name, sensor_name)

        return schemas.TaskResult(
            status="SUCCESS",
            result=schemas.TaskResultPayload(
                prediction_result=prediction_result,
                heuristic_result=heuristic_result
            )
        )