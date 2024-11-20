defmodule YOLO do
  @moduledoc false
  defdelegate load(options), to: YOLO.Model
  defdelegate detect(model, image_nx), to: YOLO.Model
  defdelegate detect(model, image_nx, options), to: YOLO.Model

  @doc """
  Adjusts bounding box coordinates and sizes from the model input dimensions back to the original image dimensions.
  `bboxes` is a list of rows, where rows are lists of 6 elements `[cx, cy, w, h, prob, class_idx]`.

  Returns a list of scaled bboxes.

    ## Example
    For a YOLOv8n model processing an image resized to 640x640, use the following to scale
    the bounding boxes back to the original 1920x1080 image:

    ```
    scale_bboxes_to_original(bboxes, {640, 640}, {1920, 1080})
    ```
  """
  @spec scale_bboxes_to_original([[float()]], YOLO.Model.shape(), YOLO.Model.shape()) :: [
          [float()]
        ]
  def scale_bboxes_to_original(bboxes, {w_input, h_input}, {w_orig, h_orig}) do
    w_ratio = w_orig / w_input
    h_ratio = h_orig / h_input

    Enum.map(bboxes, fn [cx, cy, w, h, prob, class] ->
      [
        round(cx * w_ratio),
        round(cy * h_ratio),
        round(w * w_ratio),
        round(h * h_ratio),
        prob,
        class
      ]
    end)
  end

  @doc """
  Maps `[cx, cy, w, h, prob, class_idx]` rows to `YOLO.Model.detected_object()`
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
