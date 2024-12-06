defmodule YOLO.FrameScalers do
  @moduledoc """
  Provides functions for resizing images while preserving aspect ratio and handling
  padding to reach target dimensions.


  ## Available scalers

  The following scaler implementations are currently provided:

  * `YOLO.FrameScalers.EvisionScaler` - For scaling images using `Evision`
  * `YOLO.FrameScalers.ImageScaler` - For scaling images using `Image` library
  * `YOLO.FrameScalers.NxIdentityScaler` - For when input is already an appropriately sized `Nx` tensor

  ```elixir
  # For Evision Mat images:
  YOLO.detect(model, mat, frame_scaler: YOLO.FrameScalers.EvisionScaler)

  # For Image library images:
  YOLO.detect(model, image, frame_scaler: YOLO.FrameScalers.ImageScaler)

  # For pre-sized Nx tensors:
  YOLO.detect(model, tensor, frame_scaler: YOLO.FrameScalers.NxIdentityScaler)
  ```

  """
  alias YOLO.FrameScalers.ScalingConfig

  @doc """
  Scales and pads an image to fit target dimensions.
  It's used in the model `preprocess/3` callback.

  ## Arguments
    * `image` - Input image in implementation's native format
    * `target_shape` - Target dimensions as `{width, height}` tuple
    * `resizer_module` - Module implementing the FrameScaler behaviour

  ## Returns
    * `{image_tensor, scaling_config}` where image_tensor is the processed Nx tensor
      and scaling_config is a struct containing scaling and padding information

  The returned `image_tensor` has shape `{height, width, channels}`.
  """
  @pad_value 114
  @spec fit(term(), {height :: integer(), width :: integer()}, frame_scaler :: module()) ::
          {Nx.Tensor.t(), ScalingConfig.t()}
  def fit(image, target_shape, scaler_module) do
    image_scaling =
      %{
        padding: {width_padding, height_padding}
      } = calculate_resized_shape_and_padding(image, target_shape, scaler_module)

    image_nx =
      image
      |> scaler_module.image_resize(image_scaling)
      |> scaler_module.image_to_nx()
      # {height, width, 3=_channels}
      |> Nx.pad(@pad_value, [
        {floor(height_padding), ceil(height_padding), 0},
        {floor(width_padding), ceil(width_padding), 0},
        {0, 0, 0}
      ])

    {image_nx, image_scaling}
  end

  # returns tensor and padding
  # inspired from https://github.com/hansihe/yolov8_elixir/blob/master/lib/yolo/preprocess.ex
  @spec calculate_resized_shape_and_padding(
          Nx.Tensor.t(),
          {height :: integer(), width :: integer()},
          module
        ) :: ScalingConfig.t()
  defp calculate_resized_shape_and_padding(
         image,
         {model_input_height, model_input_width},
         scaler_module
       ) do
    {image_width, image_height, _channels} = scaler_module.image_shape(image)

    width_ratio = model_input_width / image_width
    height_ratio = model_input_height / image_height
    ratio = min(width_ratio, height_ratio)

    {scaled_width, scaled_height} =
      if width_ratio < height_ratio do
        # landscape, width = model input size
        {model_input_height, ceil(image_height * ratio)}
      else
        # portrait or squared, height = model input size
        {ceil(image_width * ratio), model_input_height}
      end

    # we are going to add padding to match the model input shape
    width_padding = (model_input_width - scaled_width) / 2
    height_padding = (model_input_height - scaled_height) / 2

    %ScalingConfig{
      original_image_shape: {image_width, image_height},
      scaled_image_shape: {scaled_width, scaled_height},
      model_input_shape: {model_input_width, model_input_height},
      scale: {ratio, ratio},
      padding: {width_padding, height_padding}
    }
  end

  @doc """
  Adjusts bounding box coordinates and sizes from the model input dimensions back to the original image dimensions.
  `bboxes` is a list of rows, where rows are lists of 6 elements `[cx, cy, w, h, prob, class_idx]`.

  Returns a list of scaled bboxes.

  # Example
  For a YOLOv8n model processing an image resized to 640x640, use the following to scale
  the bounding boxes back to the original 1920x1080 image:

  ```
  scale_bboxes_to_original(bboxes, scaling_config)
  ```
  """
  @spec scale_bboxes_to_original([[float()]], ScalingConfig.t()) :: [[float()]]
  def scale_bboxes_to_original(bboxes, scaling_config) do
    # h_input = 640 - 2*140 = 360
    # w_input = 640
    %{padding: {width_padding, height_padding}, scale: {ratio, _}} = scaling_config

    Enum.map(bboxes, fn [cx, cy, w, h, prob, class] ->
      [
        round((cx - width_padding) / ratio),
        round((cy - height_padding) / ratio),
        round(w / ratio),
        round(h / ratio),
        prob,
        class
      ]
    end)
  end
end
