from pydantic import BaseModel
from typing import Optional
import enum

class Metadata(BaseModel):
    gateway_name: str
    sensor_name: str

class BaseExport(BaseModel):
    metadata: Metadata
    export_value: object

class CloudModel(BaseModel):
    tf_model_bytesize: int
    tf_model_b64: str

class SensorReading(BaseModel):
    uuid: str
    values: list[list[float]]

class InferenceLayer(int, enum.Enum):
    CLOUD = 2
    GATEWAY = 1
    SENSOR = 0

class InferenceDescriptor(BaseModel):
    inference_layer: InferenceLayer
    send_timestamp: Optional[int] = None

class SensorData(BaseModel):
    reading: SensorReading
    low_battery: bool
    inference_descriptor: InferenceDescriptor

class SensorDataExport(BaseExport):
    export_value: SensorData

class PredictionResult(BaseModel):
    gateway_name: str
    sensor_name: str
    reading_uuid: str
    low_battery: bool
    prediction: int


# --- TaskResult ---

class TaskResultPayload(BaseModel):
    prediction_result: int
    heuristic_result: Optional[int] = None

class TaskResult(BaseModel):
    status: str
    result: Optional[TaskResultPayload] = None