defmodule YOLO.Models.YOLOX do
  @moduledoc """
  YOLOX model implementation for preprocessing input images and postprocessing detections using non-maximum suppression (NMS).

  Supports YOLOX models found at [https://github.com/Megvii-BaseDetection/YOLOX](github.com/Megvii-BaseDetection/YOLOX)

  If using a YOLOX model that was exported with `--decode_in_inference`, you can set
  `decode_head: false` in the YOLO.detect/3 options.

  YOLOX-Tiny and YOLOX-Nano models use 416x416, while other models use 640x640.
  """

  @behaviour YOLO.Model

  import Nx.Defn

  @impl true
  def init(model, options) do
    {_, _, height, width} = model.shapes.input

    {grids, expanded_strides} =
      generate_grids_and_expanded_strides({width, height}, options[:p6] || false)

    %{model | model_data: %{grids: grids, expanded_strides: expanded_strides}}
  end

  @doc """
  YOLOX input doesn't need to be normalized, so we resize and convert the image to a `{batch_size, channels, height, width}` tensor.

  Tiny and Nano models use 416x416, while other models use 640x640.
  """
  @impl true
  @spec preprocess(YOLO.Model.t(), term(), Keyword.t()) :: {Nx.Tensor.t(), ScalingConfig}
  def preprocess(model, image, options) do
    frame_scaler = Keyword.fetch!(options, :frame_scaler)
    {_, _channels, height, width} = model.shapes.input
    {image_nx, image_scaling} = YOLO.FrameScalers.fit(image, {height, width}, frame_scaler)

    input_nx = do_preprocess(image_nx)

    {input_nx, image_scaling}
  end

  defnp do_preprocess(image_nx) do
    image_nx
    |> Nx.as_type({:f, 32})
    # {h, w, c} -> {c, h, w}
    |> Nx.transpose(axes: [2, 0, 1])
    # add another axis {3, 640, 640} -> {1, 3, 640, 640}
    |> Nx.new_axis(0)
  end

  @doc """
  Post-processes the model's raw output to produce a filtered list of detected objects.

  Options:
  * `decode_head` - If true, decode the output head to map predictions to input image space.
    Defaults to `true`. Can be set to `false` if using a YOLOX model that was exported with `--decode_in_inference`
  * `nms_fun` - Optional custom NMS function. Must calculate detection scores as the product of the maximum class
    probability and the objectness score.
  * `prob_threshold` - Minimum probability threshold for detections
  * `iou_threshold` - IoU threshold for non-maximum suppression
  """
  @impl true
  def postprocess(model, model_output, scaling_config, opts) do
    prob_threshold = Keyword.fetch!(opts, :prob_threshold)
    iou_threshold = Keyword.fetch!(opts, :iou_threshold)
    nms_fun = Keyword.get(opts, :nms_fun, &default_nms/3)
    decode_head? = Keyword.get(opts, :decode_head, true)

    %{grids: grids, expanded_strides: expanded_strides} = model.model_data

    model_output
    |> maybe_decode_head(grids, expanded_strides, decode_head?)
    |> nms_fun.(prob_threshold, iou_threshold)
    |> YOLO.FrameScalers.scale_bboxes_to_original(scaling_config)
  end

  defp maybe_decode_head(model_output, grids, expanded_strides, true = _decode_head?) do
    process_bboxes(model_output, grids, expanded_strides)
  end

  defp maybe_decode_head(model_output, _grids, _expanded_strides, false = _decode_head?) do
    # {1, n, 85} -> {n, 85}
    Nx.squeeze(model_output)
  end

  # YOLOX uses convolutions, so if the exported model doesn't include decoding in inference, we have to
  # apply strides to map predictions to input image space.
  # Translated from https://github.com/Megvii-BaseDetection/YOLOX/blob/d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a/yolox/utils/demo_utils.py#L139
  # Generation of grids and expanded strides is split out into `generate_grids_and_expanded_strides`
  # which is precalculated when the model is loaded, since we don't need to run it for every inference!
  defn process_bboxes(model_output, grids, expanded_strides) do
    # {1, 8400, 85} -> {8400, 85}
    model_output = Nx.squeeze(model_output, axes: [0])

    # Split outputs into slices for processing
    # Python:
    # coords = outputs[..., :2]
    # sizes = outputs[..., 2:4]
    # remainder = outputs[..., 4:]
    coords = model_output[[.., 0..1]]
    sizes = model_output[[.., 2..3]]
    remainder = model_output[[.., 4..-1//1]]

    # Align shapes for broadcasting
    # Python: grids = grids.reshape(1, -1, 2)
    #         expanded_strides = expanded_strides.reshape(1, -1, 1)
    # From {1, 8400, 2} to {8400, 2}
    grids = Nx.squeeze(grids, axes: [0])
    # From {1, 8400, 1} to {8400, 1}
    expanded_strides = Nx.squeeze(expanded_strides, axes: [0])

    # Update the coordinates and sizes using tensorized operations
    # Python:
    # outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    # outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    updated_coords = Nx.add(coords, grids) |> Nx.multiply(expanded_strides)
    updated_sizes = Nx.exp(sizes) |> Nx.multiply(expanded_strides)

    # Concatenate updated slices with the remainder of the outputs
    # Python: return outputs
    Nx.concatenate([updated_coords, updated_sizes, remainder], axis: 1)
  end

  @doc """
  Calculates the detection score for each prediction as the product of the maximum class
  probability and the objectness score.

  Adaptation of YOLO.NMS.filter_predictions/2, but calculates the correct score based on
  the product of the maximum class probability and the objectness score which differs from Ultralytics

  Removes prob_threshold filtering so that we can use Nx.Defn compilation for performance.
  """
  defn calculate_max_prob_score_per_prediction(predictions) do
    # {n, 4}
    bboxes = Nx.slice_along_axis(predictions, 0, 4, axis: 1)

    # {n, 1}
    objectness = Nx.slice_along_axis(predictions, 4, 1, axis: 1)

    # {n, 80}
    class_probs = Nx.slice_along_axis(predictions, 5, 80, axis: 1)

    # Yolox calculates detection scores as the product of maximum class probabilities and objectness score
    scores = Nx.multiply(class_probs, objectness)

    # Per row, gets the max prob and the class with that prob
    {max_prob, max_prob_class} = Nx.top_k(scores, k: 1)

    # concatenating the columns [cx, cy, w, h, prob, class]
    Nx.concatenate([bboxes, max_prob, max_prob_class], axis: 1)
  end

  # Basic ripoff of numpy.meshgrid for yolox purposes
  defnp meshgrid(opts \\ []) do
    opts = keyword!(opts, x_range: 1, y_range: 1)
    # Increase across rows
    # [[0, 1, 2, ...], [0, 1, 2, ...], ...]
    x_grid = Nx.iota({opts[:x_range], opts[:x_range]}, axis: 1)

    # Increase across columns
    # [[0, 0, 0, ...], [1, 1, 1, ...], ...]
    y_grid = Nx.iota({opts[:y_range], opts[:y_range]}, axis: 0)

    {x_grid, y_grid}
  end

  def generate_grids_and_expanded_strides({width, height}, p6 \\ false) do
    strides = if p6, do: [8, 16, 32, 64], else: [8, 16, 32]

    # Calculate feature map sizes for each stride
    # Python: hsizes = [img_size[0] // stride for stride in strides]
    #         wsizes = [img_size[1] // stride for stride in strides]
    hsizes = Enum.map(strides, fn stride -> div(height, stride) end)
    wsizes = Enum.map(strides, fn stride -> div(width, stride) end)

    # Python:
    # for hsize, wsize, stride in zip(hsizes, wsizes, strides):

    {grids, expanded_strides} =
      Enum.zip([hsizes, wsizes, strides])
      |> Enum.reduce({[], []}, fn {hsize, wsize, stride}, {grids_acc, strides_acc} ->
        # Generate meshgrid for the current stride
        # Python: xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        {xv, yv} = meshgrid(x_range: wsize, y_range: hsize)

        # Combine meshgrid components and reshape to match YOLOX output structure
        # Python: grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grid = Nx.stack([xv, yv], axis: 2) |> Nx.reshape({1, wsize * hsize, 2})

        # Create expanded strides tensor with the same shape as the grid
        # Python: expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
        expanded_stride = Nx.broadcast(Nx.tensor(stride), {1, wsize * hsize, 1})

        # Append results to accumulators
        # {Nx.concatenate([grids_acc, grid], axis: 1), Nx.concatenate([strides_acc, expanded_stride], axis: 1)}
        {grids_acc ++ [grid], strides_acc ++ [expanded_stride]}
      end)

    # Concatenate grids and expanded strides for all feature map levels
    # Python: grids = np.concatenate(grids, 1)
    #         expanded_strides = np.concatenate(expanded_strides, 1)
    grids = Nx.concatenate(grids, axis: 1)
    expanded_strides = Nx.concatenate(expanded_strides, axis: 1)

    {grids, expanded_strides}
  end

  def default_nms(model_output_nx, prob_threshold, nms_threshold) do
    model_output_nx
    |> calculate_max_prob_score_per_prediction()
    |> Nx.to_list()
    |> Stream.filter(fn [_cx, _cy, _w, _h, prob, _class] -> prob >= prob_threshold end)
    |> Enum.sort_by(fn [_cx, _cy, _w, _h, prob, _class] -> prob end, :desc)
    |> YOLO.NMS.nms(nms_threshold)
  end
end
