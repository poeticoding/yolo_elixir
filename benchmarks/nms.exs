# This benchmark compares the performance of two NMS (Non-Maximum Suppression) implementations:
# - YoloFastNMS: A NIF-based implementation written in Rust
# - YOLO.NMS: An Elixir + Nx implementation

mat = Evision.imread("benchmarks/images/traffic.jpg")
onnx_path = "models/yolo11n.onnx"
classes_path = "models/coco_classes.json"


Nx.default_backend({EMLX.Backend, device: :cpu})
Nx.Defn.default_options(compiler: EMLX)

model = YOLO.load(model_path: onnx_path, classes_path: classes_path)
{input, scaling_config} = YOLO.Models.Ultralytics.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
output = YOLO.Models.run(model, input)

fast_nms = fn ->
  YOLO.Models.Ultralytics.postprocess(
    model,
    output,
    scaling_config,
    prob_threshold: 0.25, iou_threshold: 0.45, nms_fun: &YoloFastNMS.run/2
  )
end

elixir_nms = fn ->
  YOLO.Models.Ultralytics.postprocess(
    model,
    output,
    scaling_config,
    prob_threshold: 0.25, iou_threshold: 0.45
  )
end


dbg(Nx.default_backend())

#warmup
for _ <- 1..100 do
  # fast_nms.()
  elixir_nms.()
end


Benchee.run(%{
    "YoloFastNMS" => fast_nms,

    "YOLO.NMS" => elixir_nms
  },
  time: 10,
  memory_time: 2
)
