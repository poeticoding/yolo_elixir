defmodule YOLO.NMS do
  @moduledoc """
  Elixir NMS (Non-Maximum Suppression)

  Learn more about Non-Maximum Suppression (NMS) at:
  https://builtin.com/machine-learning/non-maximum-suppression

  This implementation applies NMS independently for each output class.
  The following steps are executed for each class:
  1. Filter out all bounding boxes with maximum class probability below `prob_threshold` (default: `0.5`).
  2. Select the bounding box with the highest `prob`.
  3. Remove any remaining bounding boxes with an IoU >= `iou_threshold` (default: `0.5`).
  """

  @doc """
  Runs both `filter_predictions/2` and `nms/2`.

  1. Filters out detections with a probability below `prob_threshold` (`p < prob_threshold`)
  2. Applies Non-Maximum Suppression (NMS) for each class, discarding bounding boxes with an
     IoU exceeding the specified `iou_threshold`.

  The input `tensor` must have the shape `{8400, 84}` (transposed YOLOv8 output format).

  Returns a list of `[bbox_cx, bbox_cy, bbox_w, bbox_h, prob, class_idx]`.
  """

  import Nx.Defn

  @spec run(Nx.Tensor.t(), float(), float(), Keyword.t()) :: [[float()]]
  def run(tensor, prob_threshold, iou_threshold, opts \\ []) do
    tensor
    |> filter_predictions(prob_threshold, Keyword.take(opts, [:transpose?]))
    # the results sorted desc by probability
    |> nms(iou_threshold)
  end

  @doc """
  Filters detections based on a confidence probability threshold.

  Selects detections from `model_output` where the highest class confidence score exceeds `prob_threshold`.


  ## Arguments

    * `model_output`: Input tensor. The standard expected shape is `{detections, bbox+classes}`
      (e.g., `{8400, 84}`). The function can also handle inputs with a leading
      batch dimension (e.g., `{1, 8400, 84}`), effectively squeezing internally.
      If the input shape is `{bbox+classes, detections}` (e.g., `{84, 8400}`) or
      `{1, bbox+classes, detections}` (e.g., `{1, 84, 8400}`), set
      `transpose?: true` for internal transposition. The first 4 elements
      of the `bbox+classes` dimension must be the bounding box coordinates.

    * `prob_threshold`: Minimum confidence probability to keep a detection.

    * `opts`: Keyword list options:
      - `:transpose?` (boolean, default: `false`): If `true`, transpose the
        input `model_output` before processing.

  ## Returns

  A list of detections `[cx, cy, w, h, prob, class_idx]`, sorted descending
  by `prob`. Returns `[]` if no detections meet the threshold.
  """
  @spec filter_predictions(Nx.Tensor.t(), float(), Keyword.t()) :: [[float()]]
  def filter_predictions(model_output, prob_threshold, opts \\ []) do
    transpose? = Keyword.get(opts, :transpose?, false)

    filtered_count =
      model_output
      |> count_confident_detections(prob_threshold: prob_threshold, transpose?: transpose?)
      |> Nx.to_number()

    if filtered_count == 0 do
      []
    else
      model_output
      |> build_top_detections_tensor(count: filtered_count, transpose?: transpose?)
      |> Nx.to_list()
    end
  end

  @spec maybe_squeeze_and_transpose(Nx.Tensor.t(), Keyword.t()) :: Nx.Tensor.t()
  defnp maybe_squeeze_and_transpose(model_output, opts) do
    model_output = if Nx.rank(model_output) == 3, do: Nx.squeeze(model_output), else: model_output
    model_output = if opts[:transpose?], do: Nx.transpose(model_output), else: model_output
    model_output
  end

  # filter_predictions/2: part 1
  # count the number of detections with confidence above prob_threshold
  @spec count_confident_detections(Nx.Tensor.t(), Keyword.t()) :: Nx.Tensor.t()
  defnp count_confident_detections(model_output, opts) do
    opts = keyword!(opts, [:prob_threshold, :transpose?])
    prob_threshold = opts[:prob_threshold]

    model_output = maybe_squeeze_and_transpose(model_output, opts)

    # focusing on the class predictions
    probs = Nx.slice_along_axis(model_output, 4, Nx.axis_size(model_output, 1) - 4, axis: 1)
    max_prob = Nx.reduce_max(probs, axes: [1])

    max_prob
    |> Nx.greater_equal(prob_threshold)
    |> Nx.sum()
  end

  # filter_predictions/2: part 2
  @spec build_top_detections_tensor(Nx.Tensor.t(), Keyword.t()) :: Nx.Tensor.t()
  defnp build_top_detections_tensor(model_output, opts) do
    opts = keyword!(opts, [:count, :transpose?])
    filtered_count = opts[:count]

    model_output = maybe_squeeze_and_transpose(model_output, opts)

    {_dets_count, cols_count} = Nx.shape(model_output)
    classes_count = cols_count - 4

    # focusing on the class predictions
    probs = Nx.slice_along_axis(model_output, 4, classes_count, axis: 1)

    # getting the max probability for each row (for each detected object)
    max_prob = Nx.reduce_max(probs, axes: [1])

    sorted_prob_indices = Nx.argsort(max_prob, direction: :desc)
    filtered_indices = Nx.slice_along_axis(sorted_prob_indices, 0, filtered_count, axis: 0)

    filtered_rows = Nx.take(model_output, filtered_indices)

    filtered_probs = Nx.take(max_prob, filtered_indices) |> Nx.new_axis(-1)

    filtered_bboxes =
      filtered_rows
      |> Nx.slice_along_axis(0, 4, axis: 1)

    filtered_class_ids =
      probs
      |> Nx.argmax(axis: 1)
      |> Nx.take(filtered_indices)
      |> Nx.new_axis(-1)

    Nx.concatenate([filtered_bboxes, filtered_probs, filtered_class_ids], axis: 1)
  end

  @doc """
  Applies Non-Maximum Suppression (NMS) for each class, discarding bounding boxes with an
  IoU exceeding the specified `iou_threshold`.
  """
  def nms(bboxes, iou_threshold) do
    bboxes
    # group results by class
    |> Enum.group_by(fn [_cx, _cy, _w, _h, _prob, class] -> class end)
    # run nms for each class
    # keep the highest prob bbox and compare the IoU with rest of the bbox of the same class.
    # keep all the bbox below the given `iou_threshold`.
    |> Enum.reduce([], fn {_class, class_bboxes}, kept_bboxes ->
      # adding the results for the given class
      kept_bboxes ++ class_nms(class_bboxes, iou_threshold)
    end)
  end

  # `bboxes` are in the same class
  # returning the non-suppresed bboxes
  defp class_nms([high_prob_bbox | rest] = _class_bboxes, iou_threshold) do
    Enum.reduce(rest, [high_prob_bbox], fn bbox, kept_class_bboxes ->
      # [xc, yc, w, h, _, _]
      if iou_below_all?(bbox, kept_class_bboxes, iou_threshold) do
        # not  overlapping, keeping it.
        [bbox | kept_class_bboxes]
      else
        kept_class_bboxes
      end
    end)
  end

  defp iou_below_all?(bbox, kept_bboxes, iou_threshold) do
    Enum.all?(kept_bboxes, fn valid_bbox ->
      iou(valid_bbox, bbox) <= iou_threshold
    end)
  end

  @doc """
  IoU (Intersection over Union) measures the ratio of the area of overlap between
  two bounding boxes to the area of their union.

  IoU = A ∩ B / A U B
  """
  def iou([ax, ay, aw, ah], [bx, by, bw, bh]) do
    x1 = max(ax - aw / 2, bx - bw / 2)
    y1 = max(ay - ah / 2, by - bh / 2)
    x2 = min(ax + aw / 2, bx + bw / 2)
    y2 = min(ay + ah / 2, by + bh / 2)

    # A ∩ B
    int_w = max(x2 - x1, 0)
    int_h = max(y2 - y1, 0)
    intersection = int_w * int_h

    # A U B
    union = aw * ah + bw * bh - intersection

    intersection / union
  end

  def iou([ax, ay, aw, ah | _], [bx, by, bw, bh | _]) do
    iou([ax, ay, aw, ah], [bx, by, bw, bh])
  end
end
