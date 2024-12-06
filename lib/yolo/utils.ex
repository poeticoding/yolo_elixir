defmodule YOLO.Utils do
  @moduledoc """
  Utils to transform the model input and output.
  """

  @doc """
  Maps `[cx, cy, w, h, prob, class_idx]` rows to `YOLO.Model.detected_object()`

  # Example
  ```
  iex> YOLO.Utils.to_detected_objects([ [100, 200, 20, 30, 0.7, 2] ], yolov8n_model.classes)
  [%{
    bbox: %{cx: 100, cy: 200, w: 20, h: 30},
    prob: 0.7,
    class_idx: 2,
    class: "car"
  }]
  ```
  """
  @spec to_detected_objects([[float()]], YOLO.Model.classes()) :: [YOLO.Model.detected_object()]
  def to_detected_objects(bboxes, model_classes) do
    Enum.map(bboxes, fn [cx, cy, w, h, prob, class_idx] ->
      class_idx = round(class_idx)

      %{
        prob: prob,
        bbox: %{cx: round(cx), cy: round(cy), w: ceil(w), h: ceil(h)},
        class_idx: class_idx,
        class: Map.get(model_classes, class_idx)
      }
    end)
  end
end
