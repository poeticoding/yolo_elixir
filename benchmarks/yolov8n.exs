eps = case System.argv() do
  ["--eps", "directml"] -> [:directml]
  ["--eps", "coreml"] -> [:coreml]
  ["--eps", "cuda"] -> [:cuda]
  ["--eps", "tensorrt"] -> [:tensorrt]
  _ -> [:cpu]
end

images = [
  "dog", "eagle", "giraffe", "horses", "kite", "person", "scream", "traffic"
]
|> Enum.map(&Path.join("benchmarks/images", "#{&1}.jpg"))
|> Enum.map(&Evision.imread/1)

onnx_path = "models/yolov8n.onnx"
classes_path = "models/yolov8n_classes.json"


model = YOLO.load(model_path: onnx_path, classes_path: classes_path, eps: eps)

dbg(Nx.default_backend())
Benchee.run(%{
    "detect/3 with FastNMS on #{List.first(eps)}" => {
      fn mat ->
        YOLO.detect(model, mat, nms_fun: &YoloFastNMS.run/3)
      end,
      before_each: fn _ ->
        Enum.random(images)
      end
    },
    "run/3 on #{List.first(eps)}" => {
      fn input -> YOLO.Models.run(model, input) end,
      before_each: fn _ ->
        mat = Enum.random(images)
        {input, _scaling_config} = YOLO.Models.YoloV8.preprocess(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)
        input
      end
    }

  },
  time: 10,
  memory_time: 2
)
