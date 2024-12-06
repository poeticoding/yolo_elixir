# This benchmark compares the performance of two NMS (Non-Maximum Suppression) implementations:
# - YoloFastNMS: A NIF-based implementation written in Rust
# - YOLO.NMS: An Elixir + Nx implementation

mat = Evision.imread("benchmarks/images/traffic.jpg")
onnx_path = "models/yolov8n.onnx"
classes_path = "models/yolov8n_classes.json"

model = YOLO.load(model_path: onnx_path, classes_path: classes_path)
{input, scaling_config} = YOLO.Models.YoloV8.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
output = YOLO.Models.run(model, input)

dbg(Nx.default_backend())
Benchee.run(%{
    "YoloFastNMS" => fn ->
      YOLO.Models.YoloV8.postprocess(
        model,
        output,
        scaling_config,
        prob_threshold: 0.25, iou_threshold: 0.45, nms_fun: &YoloFastNMS.run/3
      )
    end,

    "YOLO.NMS" => fn ->
      YOLO.Models.YoloV8.postprocess(
        model,
        output,
        scaling_config,
        prob_threshold: 0.25, iou_threshold: 0.45
      )
    end
  },
  time: 10,
  memory_time: 2
)
