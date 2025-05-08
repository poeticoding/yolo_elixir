defmodule YOLO do
  @moduledoc false
  defdelegate load(options), to: YOLO.Models
  defdelegate detect(model, image), to: YOLO.Models
  defdelegate detect(model, image, options), to: YOLO.Models

  defdelegate to_detected_objects(bboxes, model_classes), to: YOLO.Utils
end
