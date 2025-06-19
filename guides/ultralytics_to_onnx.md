# Ultralytics to ONNX

Download the PyTorch model from the [Ultralytics assets repository](https://github.com/ultralytics/assets/releases/tag/v8.3.0). At the moment, the latest version is YOLOv11, available in various sizes: `yolov11n.pt` (nano), `yolov11s.pt` (small), `yolov11m.pt` (medium), etc. For detailed specifications of each model variant, visit the [Ultralytics documentation](https://docs.ultralytics.com/tasks/detect/#models).


I've prepared a python [`ultralytics_to_onnx.py` script](python/ultralytics_to_onnx.py) to easily download and convert Ultralytics `pt` models to the right `onnx` format.

To run it, first, you need to install the dependencies (`requests` and `ultralytics`)
```bash
pip install -r python/requirements.txt
```

Then, run the script by specifying the model size, such as `n`:

```bash
python python/ultralytics_to_onnx.py yolo11n
```

The script will download the `.pt` model and generate two files:

* `models/yolo11n.onnx`: the Yolo11n model with weights
* `models/yolo11n_classes.json`: the list of object classes