from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

subscription_id = ""
resource_group = ""
workspace = ""


ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
