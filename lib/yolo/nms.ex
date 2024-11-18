defmodule YOLO.NMS do
  @moduledoc """
  Elixir NMS (Non-Max Suppression)
  You can find more on NMS on https://builtin.com/machine-learning/non-maximum-suppression

  This implementation runs NMS independently on each of the output classes.
  Here's the step that are run for each class:
  1. Discard all the bboxes with `prob <= filter_threshold` (`0.5` by default).
  2. Pick box with largest `prob`.
  3. Discard any remaining box with IoU >= `iou_threshold` (`0.5` by default).
  """
  @spec nms(Nx.Tensor.t(), float(), float()) :: [[float()]]
  def nms(tensor, prob_threshold, nms_iou_threshold) do
    # `tensor` rows are the results sorted desc by probability.
    tensor
    |> filter_predictions(prob_threshold)
    |> do_nms(nms_iou_threshold)
  end

  defp do_nms(bboxes, nms_iou_threshold) do
    bboxes
    # group by classes
    |> Enum.group_by(fn [_cx, _cy, _w, _h, _prob, class] -> class end)
    # run nms for each class
    |> Enum.reduce([], fn {_class, [high_prob_bbox | rest] = _class_bboxes}, kept_boxes ->
      kept_class_boxes =
        Enum.reduce(rest, [high_prob_bbox], fn bbox, kept_class_boxes ->
          # [xc, yc, w, h, _, _]
          if Enum.all?(kept_class_boxes, fn valid_bbox ->
               iou(valid_bbox, bbox) <= nms_iou_threshold
             end) do
            [bbox | kept_class_boxes]
          else
            kept_class_boxes
          end
        end)

      [kept_class_boxes | kept_boxes]
    end)
  end

  @doc """
  Keeps only the detections with a probability higher than `:prob_threshold`
  `tensor` must be `{8400, 84}` (transposed YoloV8n output shape)
  returns a list of `[bbox_cx, bbox_cy, bbox_w, bbox_h, prob, class_idx]`
  implementation inspired by Hans Elias B. Josephsen's talk (12:06 you can see the filter function)
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
