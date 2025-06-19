[eps, model_path] = System.argv()

eps = case eps do
  "directml" -> [:directml]
  "coreml" -> [:coreml]
  "cuda" -> [:cuda]
  "tensorrt" -> [:tensorrt]
  _ -> [:cpu]
end


mat = Evision.imread("benchmarks/images/traffic.jpg")

classes_path = "models/coco_classes.json"

model = YOLO.load(model_path: model_path, classes_path: classes_path, eps: eps)

dbg(Nx.default_backend())

### warmup
yolo_pipeline = fn mat ->
  {input, scaling_config} = YOLO.Models.Ultralytics.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
  model_input = Nx.backend_copy(input, EXLA.Backend)
  output = YOLO.Models.run(model, input)
  detections = YOLO.Models.Ultralytics.postprocess(model, output, scaling_config, prob_threshold: 0.25, iou_threshold: 0.45)
  {model_input, scaling_config, output, detections}
end

for _ <- 1..50 do
  yolo_pipeline.(mat)
end

{model_input, scaling_config, model_output, _detections} = yolo_pipeline.(mat)

Benchee.run(%{
    "preprocess" => fn ->
      YOLO.Models.Ultralytics.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
    end,
    "run" => {
      fn input -> YOLO.Models.run(model, input) end,
      before_each: fn _ ->
        Nx.backend_copy(model_input, EXLA.Backend)
      end
    },
    "postprocess" => fn ->
      YOLO.Models.Ultralytics.postprocess(model, model_output, scaling_config, prob_threshold: 0.25, iou_threshold: 0.45)
    end
  },
  time: 10,
  memory_time: 2
)
