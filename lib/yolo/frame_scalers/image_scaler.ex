if Code.ensure_loaded?(Image) do
  defmodule YOLO.FrameScalers.ImageScaler do
    @moduledoc """
    Implementation of the YOLO.FrameScaler behaviour for Vix.Vips.Image objects (`:image` library).
    """
    @behaviour YOLO.FrameScaler

    @impl true
    def image_shape(%Vix.Vips.Image{} = image) do
      # {width, height, channels}
      Image.shape(image)
    end

    @impl true
    def image_resize(image, scale_config) do
      {scale_w, _scale_h} = scale_config.scale
      Image.resize!(image, scale_w)
    end

    @impl true
    def image_to_nx(image) do
      {backend, _} = Nx.default_backend()
      Image.to_nx!(image, backend: backend)
    end
  end
end
