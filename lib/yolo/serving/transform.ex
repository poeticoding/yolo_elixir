defmodule Yolo.Serving.Transform do
  @moduledoc """

  """
  def call(bboxes, model_classes) do
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
