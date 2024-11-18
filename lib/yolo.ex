defmodule YOLO do
  @moduledoc false
  defdelegate load(options), to: YOLO.Model
  defdelegate detect(model, image_nx), to: YOLO.Model
  defdelegate detect(model, image_nx, options), to: YOLO.Model
end
