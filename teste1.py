# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="tiPc1eJAdtgHNw3NwPdZ"
)

# infer on a local image
result = CLIENT.infer("download (3).jpeg", model_id="macas-kyohn/2")
