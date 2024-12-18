defmodule Yolo.Serving.Detect do
  @moduledoc """
  This module handles running object detection on images using Nx.Serving.
  """

  alias Yolo.Serving.Model
  alias Yolo.Serving.Transform

  @default_detect_options [
    prob_threshold: 0.25,
    iou_threshold: 0.45,
    frame_scaler: YOLO.FrameScalers.EvisionScaler
  ]

  @doc """
  Performs object detection on an image using a loaded YOLO model.

  ## Arguments
  * `serving` - A loaded `YOLO.Model.t()` struct
  * `image` - Input image in the format expected by the frame scaler (e.g. `Evision.Mat`)
  * `opts` - Detection options

  ## Options
  * `prob_threshold` - Minimum probability threshold for detections (default: `#{@default_detect_options[:prob_threshold]}`)
  * `iou_threshold` - IoU threshold for non-maximum suppression (default: `#{@default_detect_options[:iou_threshold]}`)
  * `frame_scaler` - Module implementing `YOLO.FrameScaler` behaviour (default: `YOLO.FrameScalers.EvisionScaler`)

  ## Returns
  A list of detections, where each detection is a list `[cx, cy, w, h, prob, class_idx]`:
  * `cx`, `cy` - Center coordinates of bounding box
  * `w`, `h` - Width and height of bounding box
  * `prob` - Detection probability
  * `class_idx` - Class index

  The output can be converted to structured maps using `to_detected_objects/1`.

  ## Example
      model
      |> YOLO.Model.detect(image, prob_threshold: 0.5)
      |> YOLO.Model.to_detected_objects()
  """
  def call(image, serving, opts \\ []) do
    opts = Keyword.merge(@default_detect_options, opts)

    %{model_impl: model_impl} = model = Model.get_model(serving)

    {input_nx, scaling_config} = model_impl.preprocess(model, image, opts)
    image_batch = Nx.Batch.stack([{input_nx[0]}])

    {output_nx} = Nx.Serving.batched_run(serving, image_batch)

    model
    |> model_impl.postprocess(output_nx, scaling_config, opts)
    |> Transform.call(model.classes)
  end
end
