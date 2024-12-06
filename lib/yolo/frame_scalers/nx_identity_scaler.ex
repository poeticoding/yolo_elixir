defmodule YOLO.FrameScalers.NxIdentityScaler do
  @moduledoc """
  `FrameScaler` for when the input image is already a tensor with shape {height, width, 3}
  and no scaling is needed because width and height already match the model input size.
  This is useful when calling `YOLO.detect/3` with a tensor that already matches the
  expected dimensions.
  """
  @behaviour YOLO.FrameScaler

  @impl true
  def image_shape(tensor) do
    Nx.shape(tensor)
  end

  @impl true
  def image_resize(tensor, _) do
    tensor
  end

  @impl true
  def image_to_nx(tensor) do
    tensor
  end
end
