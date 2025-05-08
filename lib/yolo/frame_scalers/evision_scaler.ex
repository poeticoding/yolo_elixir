if Code.ensure_loaded?(Evision) do
  defmodule YOLO.FrameScalers.EvisionScaler do
    @moduledoc """
    Implementation of the `YOLO.FrameScaler` behaviour for `Evision.Mat` objects (`:evision` library).
    """
    @behaviour YOLO.FrameScaler

    @impl true
    def image_shape(%Evision.Mat{} = mat) do
      {height, width, channels} = Evision.Mat.shape(mat)
      {width, height, channels}
    end

    @impl true
    def image_resize(image, %{scaled_image_shape: {width, height}}) do
      Evision.resize(image, {width, height})
    end

    @impl true
    def image_to_nx(image) do
      {backend, _} = Nx.default_backend()
      Evision.Mat.to_nx(image, backend)
    end
  end
end
