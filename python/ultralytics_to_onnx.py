from ultralytics import YOLO
from pathlib import Path

import requests
import json
import os
import sys


MODEL_NAME = sys.argv[1]
IMAGE_SIZE = 640
PT_MODEL_URL = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{MODEL_NAME}.pt"

export_dir = "models"


pt_file = f"{export_dir}/{MODEL_NAME}.pt"
model_file = f"{export_dir}/{MODEL_NAME}.onnx"
classes_file = f"{export_dir}/{MODEL_NAME}_classes.json"


# making `export_dir`
try:
    os.mkdir(export_dir)
except OSError as error:
    print(export_dir + ": directory already created")

def download_pt_model(url, path):
  print(f"Downloading {MODEL_NAME}.pt file")
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(path, 'wb') as fp:
      for chunk in r.iter_content(chunk_size=8192): 
        fp.write(chunk)
            
  return path


model_path = download_pt_model(PT_MODEL_URL, pt_file)

# Load your pre-trained model
model = YOLO(pt_file)

# Export the models
model.export(format='onnx', imgsz=IMAGE_SIZE, opset=12)

# Export the categories
with open(classes_file, "w") as f:
  data = [model.names[idx] for idx in model.names]
  json.dump(data, f)
