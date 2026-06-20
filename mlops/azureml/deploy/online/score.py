import json
import os
import mlflow
import pandas as pd
from pathlib import Path

def init():
    global model
    model_path = Path(os.environ["AZUREML_MODEL_DIR"]) / "taxi-model"
    model = mlflow.sklearn.load_model(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    input_df = pd.DataFrame(data["input_data"])
    predictions = model.predict(input_df)
    return predictions.tolist()
