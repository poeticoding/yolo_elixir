# Nx.default_backend(EXLA.Backend)

images = [
  "dog", "eagle", "giraffe", "horses", "kite", "person", "scream", "traffic"
]
|> Enum.map(&Path.join("benchmarks/images", "#{&1}.jpg"))
|> Enum.map(&Evision.imread/1)

onnx_path = "models/yolov8n.onnx"
classes_path = "models/yolov8n_classes.json"

model = YOLO.load(model_path: onnx_path, classes_path: classes_path)

dbg(Nx.default_backend())
Benchee.run(%{
    "preprocess" => {
      fn mat ->
        YOLO.Models.YoloV8.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
      end,
      before_each: fn _ ->
        Enum.random(images)
      end
    },
    "run" => {
      fn input -> YOLO.Models.run(model, input) end,
      before_each: fn _ ->
        mat = Enum.random(images)
        {input, _scaling_config} = YOLO.Models.YoloV8.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
        input
      end
    },
    "postprocess FastNMS" => {
      fn {output, scaling_config} ->
        YOLO.Models.YoloV8.postprocess(model, output, scaling_config, prob_threshold: 0.25, iou_threshold: 0.45, nms_fun: &YoloFastNMS.run/3)
      end,
      before_each: fn _ ->
        mat = Enum.random(images)
        {input, scaling_config} = YOLO.Models.YoloV8.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
        output = YOLO.Models.run(model, input)
        {output, scaling_config}
      end
    }
  },
  time: 10,
  memory_time: 2
)
