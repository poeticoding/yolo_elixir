defmodule YOLO do
  @moduledoc """
  This module provides the main entry point and library context for YOLO object detection in Elixir.
  It delegates to the underlying model and utility modules for most functionality.
  """

  defdelegate load(options), to: YOLO.Models
  defdelegate detect(model, image), to: YOLO.Models
  defdelegate detect(model, image, options), to: YOLO.Models

  defdelegate to_detected_objects(bboxes, model_classes), to: YOLO.Utils
end
