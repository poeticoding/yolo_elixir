defmodule YOLO do

  defdelegate load(options), to: YOLO.Model
  defdelegate detect(model, image_nx), to: YOLO.Model
  defdelegate detect(model, image_nx, options), to: YOLO.Model
end
