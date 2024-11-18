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
  * `prob_threshold` by default is `0.5`.
  * `nms_iou_threshold` by default is `0.5`.
  """
  @impl true
  @spec postprocess(Nx.Tensor.t(), Keyword.t()) :: [YOLO.Model.detected_object()]
  def postprocess(model_output_nx, opts) do
    prob_threshold = Keyword.fetch!(opts, :prob_threshold)
    nms_iou_threshold = Keyword.fetch!(opts, :nms_iou_threshold)

    model_output_nx
    # from {1, 84, 8400} to {8400, 84}
    |> postprocess_transpose()
    |> YOLO.NMS.nms(prob_threshold, nms_iou_threshold)
  end

  defp postprocess_transpose(output_nx) do
    output_nx
    # from {1, 84, 8400} to {84, 8400}
    |> Nx.reshape({84, 8400})
    # transpose, 8400 rows are the detected objects, 84 bbox and probs
    |> Nx.transpose(axes: [1, 0])
  end
end
