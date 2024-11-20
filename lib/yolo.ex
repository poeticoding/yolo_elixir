defmodule YOLO do
  @moduledoc false

  # load and use model
  defdelegate load(options), to: YOLO.Model
  defdelegate detect(model, image_nx), to: YOLO.Model
  defdelegate detect(model, image_nx, options), to: YOLO.Model

  # utils
  defdelegate scale_bboxes_to_original(bboxes, model_shape, original_shape), to: YOLO.Utils
  defdelegate to_detected_objects(bboxes, model_classes), to: YOLO.Utils
end
