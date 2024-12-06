

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
    "detect/3 with FastNMS" => {
      fn mat ->
        YOLO.detect(model, mat, nms_fun: &YoloFastNMS.run/3)
      end,
      before_each: fn _ ->
        Enum.random(images)
      end
    }
  },
  time: 10,
  memory_time: 2
)
