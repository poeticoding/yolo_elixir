# Under the Hood
Let's see how `YOLO.detect/3` works.

## Load Yolo11n Model
Loads the *Yolo11n* model using the `model_path` and `classes_path`. Optionally, specify `model_impl`, which defaults to `YOLO.Models.Ultralytics`.

```elixir
model = YOLO.load([
  model_path: "models/yolo11n.onnx", 
  classes_path: "models/yolo11n_classes.json"
])
```

## Preprocessing

```elixir
mat = Evision.imread(image_path)

{input_tensor, scaling_config} = YOLO.Models.Ultralytics.preprocess(model, mat, [frame_scaler: YOLO.FrameScalers.EvisionScaler])
```

Before running object detection, the input image needs to be preprocessed to match the model's expected input format. The preprocessing steps are:

1. **Resize and Pad Image to 640x640**
   - The image is resized while preserving aspect ratio to fit within 640x640 pixels
   - Any remaining space is padded with gray color (value 114) to reach exactly 640x640
   - This is handled by the `FrameScaler` behaviour and its implementations

2. **Convert to Normalized Tensor**
   - The image is converted to an Nx tensor with shape `{1, 3, 640, 640}`
   - Pixel values are normalized from `0-255` to `0.0-1.0` range
   - The channels are reordered from `RGB` to the model's expected format (`BGR` in this case)

The `FrameScaler` behaviour provides a consistent interface for handling different image formats:

- `EvisionScaler` - For OpenCV Mat images from Evision
- `ImageScaler` - For images using the Image library  
- `NxIdentityScaler` - For ready to use Nx tensors


## Run Object Detection
Then run the detection by passing the `model` and the image tensor `input_tensor`.

```elixir
# input_tensor {1, 3, 640, 640}
output_tensor = YOLO.Models.run(model, input_tensor)
# output_tensor {1, 84, 8400}
```

You can also adjust detection thresholds (`iou_threshold` and `prob_threshold`, which both default to `0.45` and `0.25` respectively) using the third argument.


## Postprocessing
```elixir
result_rows = YOLO.Models.Ultralytics.postprocess(model, output_tensor, scaling_config, opts)
```
where `result_rows` is a list of lists, where each inner list represents a detected object with 6 elements:

```elixir
[
  [cx, cy, w, h, prob, class_idx],
  ...
]
```

The model's raw output needs to be post-processed to extract meaningful detections. For YOLOv8n, the `output_tensor` has shape `{1, 84, 8400}` where:

- 84 represents 4 bbox coordinates + 80 class probabilities
- 8400 represents the number of candidate detections

The postprocessing steps are:

1. **Filter Low Probability Detections**
   - Each detection has probabilities for all classes
   - Only keep detections where max class probability exceeds `prob_threshold` (default 0.25)

2. **Non-Maximum Suppression (NMS)**
   - Remove overlapping boxes for the same object
   - For each class, compare boxes using Intersection over Union (IoU)
   - If IoU > `iou_threshold` (default 0.45), keep only highest probability box
   - This prevents multiple detections of the same object

3. **Scale Coordinates**
   - The detected coordinates are based on the model's 640x640 input
   - Use the `scaling_config` from preprocessing to map back to original image size
   - This accounts for any resizing/padding done during preprocessing


## Convert Detections to Structured Maps
Finally, convert the raw detection results into structured maps containing bounding box coordinates, class labels, and probabilities:
```elixir
iex> YOLO.to_detected_objects(result_rows, model.classes)
[
  %{
    class: "person",
    prob: 0.57,
    bbox: %{h: 126, w: 70, cx: 700, cy: 570},
    class_idx: 0
  },
  ...
]
```

## Render results on the image

To visualize the detection results on the image, we can use the `KinoYOLO.Draw.draw_detected_objects/2` function. This function takes an `Image` and a list of detected objects, and returns a new image with bounding boxes and labels drawn on it.