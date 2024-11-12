defmodule YOLO.Models.YoloV8 do
  @behaviour YOLO.Model
  @doc """
  Prepares the image to be given as an input of the model.
  The `image_nx` tensor must be a `{640, 640, 3}`

  Transposes and reshapes `image_nx` returning a `{1, 3, 640, 640}` tensor.
  """
  @impl true
  @spec preprocess(Nx.Tensor.t()) :: Nx.Tensor.t()
  def preprocess(image_nx) do
    validate_shape!(image_nx.shape)

    # axis 0 is height
    # axis 1 is width
    # axis 2 is rgb channels
    image_nx
    # converting to float
    |> Nx.as_type({:f, 32})
    # normalizing (values between 0 and 1)
    |> Nx.divide(255)
    # transpose to a `{3, 640, 640}`
    |> Nx.transpose(axes: [2, 0, 1])
    # add another axis {3, 640, 640} -> {1, 3, 640, 640}
    |> Nx.new_axis(0)
  end

  defp validate_shape!({640, 640, 3}), do: true

  defp validate_shape!(shape) do
    raise ArgumentError,
          "the input image tensor must be of shape `{640, 640, 3}` instead of " <> inspect(shape)
  end


  @doc """
  Evaluates the model's output returning a filtered list of detected objects.
  Options:
  * `filter_threshold` by default is `0.5`.
  * `nms_threshold` by default is `0.5`.
  """
  @impl true
  def postprocess(model_output_nx, opts) do
    filter_threshold = Keyword.fetch!(opts, :filter_threshold)
    # _nms_threshold = Keyword.fetch!(opts, :nms_threshold)

    model_output_nx
    # from {1, 84, 8400} to {8400, 84}
    |> postprocess_transpose()
    |> postprocess_filter_predictions(filter_threshold)

  end

  defp postprocess_transpose(output_nx) do
    output_nx
    # from {1, 84, 8400} to {84, 8400}
    |> Nx.reshape({84, 8400})
    # transpose, 8400 rows are the detected objects, 84 bbox and probs
    |> Nx.transpose(axes: [1, 0])
  end

  # keeps only the detections with a probability higher than `:filter_threshold`
  # `output_nx` is `{8400, 84}`
  # returns a list of `[bbox_cx, bbox_cy, bbox_w, bbox_h, prob, class_idx]`
  # implementation inspired by Hans Elias B. Josephsen's talk (12:06 you can see the filter function)
  # https://youtu.be/OsxGB6MbA8o?si=cOv9HFAQrWJ8z_D8&t=726
  @spec postprocess_filter_predictions(Nx.Tensor.t(), float()) :: [[float()]]
  defp postprocess_filter_predictions(output_nx, filter_threshold) do
    bboxes = Nx.slice(output_nx, [0, 0], [8400, 4])
    # focusing on the class predictions
    probs = Nx.slice(output_nx, [0, 4], [8400, 80])
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
    Enum.take_while(Nx.to_list(detected_objects), fn [_cx, _cy, _w, _h, prob, _class] -> prob >= filter_threshold end)
  end

end
