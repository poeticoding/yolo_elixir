defmodule YOLO.FrameScaler do
  @moduledoc """
  Behaviour module defining callbacks for part of the image preprocessing required by YOLO models.

  The callbacks in this module handle three key preprocessing steps:

  1. Getting the input image dimensions to calculate proper scaling
  2. Resizing the image while preserving aspect ratio
  3. Converting the image to an Nx tensor in the correct format

  The `YOLO.FrameScalers.fit/3` function uses the callbacks to:
  - Scale the image to fit the target dimensions while maintaining aspect ratio
  - Pad any remaining space with gray color (value 114).
    [Why use 114,114,114?](https://github.com/ultralytics/ultralytics/issues/14584)

  Implementations of this behaviour provide the interface between different image
  processing libraries and the common YOLO preprocessing requirements.
  """

  alias YOLO.FrameScalers.ScalingConfig

  @doc """
  Returns the shape of the input image as a tuple of `{width, height, channels}`.
  The shape information is used to properly scale and pad images while maintaining aspect ratio.
  """
  @callback image_shape(image :: term()) ::
              {width :: integer(), height :: integer(), channels :: integer()}

  @doc """
  Resizes the input image according to the scaling configuration.

  The scaling configuration contains the target dimensions and scaling factors needed to resize
  the image while preserving aspect ratio. The resize is done in the implementation's native
  format before later conversion to an Nx tensor via `image_to_nx/1`.

  ## Arguments
    * `image` - Input image in implementation's native format
    * `scaling_config` - `ScalingConfig` struct containing target dimensions and scale factors

  ## Returns
    * Resized image in implementation's native format
  """
  @callback image_resize(image :: term(), scaling_config :: ScalingConfig.t()) :: term()

  @doc """
  Converts the implementation's native image format to an Nx tensor.
  The tensor should have shape `{height, width, channels}` where channels is typically 3 for RGB images.
  """
  @callback image_to_nx(image :: term()) :: Nx.Tensor.t()
end
