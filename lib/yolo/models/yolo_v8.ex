defmodule YOLO.Models.YoloV8 do
  @moduledoc """
  YoloV8 model implementation that delegates to the Ultralytics module.
  This module is deprecated and will be removed in the future.
  """
  alias YOLO.Models.Ultralytics

  @deprecated "Use YOLO.Models.Ultralytics.preprocess/3 instead"
  defdelegate preprocess(model, image, options), to: Ultralytics

  @deprecated "Use YOLO.Models.Ultralytics.postprocess/4 instead"
  defdelegate postprocess(model, model_output, scaling_config, opts), to: Ultralytics
end
