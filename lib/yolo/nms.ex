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
  @spec run(Nx.Tensor.t(), float(), float()) :: [[float()]]
  def run(tensor, prob_threshold, iou_threshold) do
    tensor
    |> filter_predictions(prob_threshold)
    # the results sorted desc by probability
    |> nms(iou_threshold)
  end

  @doc """
  Filters detections, keeping only those with a probability higher than `:prob_threshold`.

  The input `tensor` must have the shape `{8400, 84}` (transposed YOLOv8 output format).

  Returns a list of `[bbox_cx, bbox_cy, bbox_w, bbox_h, prob, class_idx]`.

  This implementation is inspired by Hans Elias B. Josephsen's talk (see the filter function at 12:06):
  https://youtu.be/OsxGB6MbA8o?t=726
  """
  @spec filter_predictions(Nx.Tensor.t(), float()) :: [[float()]]
  def filter_predictions(tensor, prob_threshold) do
    bboxes = Nx.slice(tensor, [0, 0], [8400, 4])
    # focusing on the class predictions
    probs = Nx.slice(tensor, [0, 4], [8400, 80])
    # getting the max probability for each row (for each detected object)
    max_prob = Nx.reduce_max(probs, axes: [1])
    # for each row (each detected object) get the class index with max prob
    max_prob_class = Nx.argmax(probs, axis: 1)

    # returning the indices of a descending ordered `max_prob` tensor
    sorted_idx = Nx.argsort(max_prob, direction: :desc)

    # concatenating the columns [cx, cy, w, h, prob, class] and getting the rows in sorted desc order
    detected_objects =
      Nx.concatenate([bboxes, Nx.new_axis(max_prob, 1), Nx.new_axis(max_prob_class, 1)], axis: 1)
      |> Nx.take(sorted_idx)

    # taking only the rows above the given probability threshold
    Enum.take_while(Nx.to_list(detected_objects), fn [_cx, _cy, _w, _h, prob, _class] ->
      prob >= prob_threshold
    end)
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
