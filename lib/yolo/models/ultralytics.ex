defmodule YOLO.Models.Ultralytics do
  @moduledoc """
  Ultralytics model implementation for preprocessing input images
  and postprocessing detections using non-maximum suppression (NMS).

  Supports YOLOv8 and YOLOv11 models trained on the COCO dataset (80 classes).
  """

  @behaviour YOLO.Model
  alias YOLO.FrameScalers.ScalingConfig

  @doc """
  Preprocesses an input image to match the model's required format.

  The preprocessing steps are:
  1. Scales and pads the image to match model input dimensions while preserving aspect ratio
  2. Converts RGB to BGR color format (Ultralytics models expect BGR input)
  3. Normalizes pixel values to [0,1] range by dividing by 255
  4. Transposes dimensions to match model's expected format
  5. Adds batch dimension

  ## Arguments
    * `model` - YOLO.Model struct containing model metadata
    * `image` - Input image in implementation's native format
    * `options` - Keyword list of options including `:frame_scaler` module

  ## Returns
    * `{input_tensor, scaling_config}` tuple where:
      * `input_tensor` has shape {1, 3, height, width}
      * `scaling_config` contains scaling/padding info for postprocessing
  """
  @impl true
  @spec preprocess(YOLO.Model.t(), term(), Keyword.t()) :: {Nx.Tensor.t(), ScalingConfig}
  def preprocess(model, image, options) do
    frame_scaler = Keyword.fetch!(options, :frame_scaler)
    {_, _channels, height, width} = model.shapes.input
    {image_nx, image_scaling} = YOLO.FrameScalers.fit(image, {height, width}, frame_scaler)

    # image_nx
    # axis 0 is height
    # axis 1 is width
    # axis 2 is rgb channels

    input_nx =
      image_nx

      # RGB to BGR
      |> Nx.reverse(axes: [2])
      |> Nx.as_type({:f, 32})
      # normalizing (values between 0 and 1)
      |> Nx.divide(255)
      # transpose to a `{3, 640, 640}`
      |> Nx.transpose(axes: [2, 0, 1])
      # add another axis {3, 640, 640} -> {1, 3, 640, 640}
      |> Nx.new_axis(0)

    {input_nx, image_scaling}
  end

  @doc """
  Post-processes the model's raw output to produce a filtered list of detected objects.

  The raw output tensor has shape {1, 84, 8400} where:
  - First dimension (1) is the batch size
  - Second dimension (84) contains 4 bbox coordinates + 80 class probabilities per detection
  - Third dimension (8400) is the number of candidate detections

  The processing steps are:
  1. Reshapes and transposes output to {8400, 84} format
  2. Applies non-maximum suppression (NMS) to filter overlapping detections
  3. Scales bounding boxes back to original image dimensions

  ## Arguments
    * `model` - YOLO.Model struct containing model metadata
    * `model_output` - Raw output tensor from model inference `{1, 84, 8400}`
    * `scaling_config` - Scaling configuration from preprocessing step
    * `opts` - Keyword list of options:
      * `prob_threshold` - Minimum probability threshold for detections
      * `iou_threshold` - IoU threshold for non-maximum suppression
      * `nms_fun` - Optional custom NMS function (defaults to `YOLO.NMS.run/3`)

  ## Returns
  List of detections where each detection is a list [cx, cy, w, h, prob, class_idx]:
    * cx, cy - Center coordinates of bounding box
    * w, h - Width and height of bounding box
    * prob - Detection probability
    * class_idx - Class index (0-79 corresponding to COCO classes)
  """
  @impl true
  @spec postprocess(
          model :: YOLO.Model.t(),
          model_output :: Nx.Tensor.t(),
          scaling_config :: ScalingConfig.t(),
          opts :: Keyword.t()
        ) :: [[float()]]
  def postprocess(_model, model_output_nx, scaling_config, opts) do
    prob_threshold = Keyword.fetch!(opts, :prob_threshold)
    iou_threshold = Keyword.fetch!(opts, :iou_threshold)
    nms_fun = Keyword.get(opts, :nms_fun, &default_nms/3)

    model_output_nx
    |> nms_fun.(prob_threshold, iou_threshold)
    |> YOLO.FrameScalers.scale_bboxes_to_original(scaling_config)
  end

  @impl true
  def precalculate(_model_ref, _shapes, _options), do: nil

  defp default_nms(model_output_nx, prob_threshold, iou_threshold) do
    model_output_nx
    |> postprocess_transpose()
    |> YOLO.NMS.run(prob_threshold, iou_threshold)
  end

  defp postprocess_transpose(output_nx) do
    output_nx
    # from {1, 84, 8400} to {84, 8400}
    |> Nx.reshape({84, 8400})
    # transpose, 8400 rows are the detected objects, 84 bbox and probs
    |> Nx.transpose(axes: [1, 0])
  end
end
