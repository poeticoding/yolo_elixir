defmodule YOLO.Models do
  @moduledoc """
  This module handles loading YOLO models and running object detection on images.

  The `YOLO.Model` behaviour can be implemented for various YOLO variants.

  ## Main Functions

  The key functions you'll use are:

  - `YOLO.Models.load/1`: Loads a YOLO model with required options
    ```elixir
    YOLO.Models.load(model_path: "path/to/model.onnx",
                    classes_path: "path/to/classes.json",
                    model_impl: YOLO.Models.YOLOX)
    ```

  - `YOLO.Models.detect/3`: Runs object detection on an image
    ```elixir
    YOLO.Models.detect(model, image, prob_threshold: 0.5)
    ```
  """
  require Logger

  @default_load_options [
    model_impl: YOLO.Models.YOLOX,
    eps: [:cpu],
    json_decoder: &:json.decode/1
  ]
  @default_detect_options [
    prob_threshold: 0.25,
    iou_threshold: 0.45,
    frame_scaler: YOLO.FrameScalers.EvisionScaler
  ]

  @doc """
  Loads a YOLO model from an ONNX file.

  ## Required Options
  * `model_path` - Path to the `.onnx` model file
  * `classes_path` - Path to the `.json` file containing class labels

  ## Optional Options
  * `model_impl` - Module implementing the `YOLO.Model` behaviour (default: YOLO.Models.YOLOX)

  * `eps` - List of execution providers to pass to Ortex (e.g. `[:coreml]`, `[:cuda]`, `[:tensorrt]`, `[:directml]`), default: `[:cpu]`
  * `json_decoder` - Function to decode JSON strings (default: `&:json.decode/1`)

  ## Returns
  A `YOLO.Model.t()` struct containing:
  * `ref` - Reference to the loaded ONNX model
  * `model_impl` - The module implementing the model version
  * `classes` - Map of class indices to labels
  * `shapes` - Input/output tensor shapes
  * `model_data` - Model-specific data

  ## Example
    ```elixir
    YOLO.Model.load(
      model_path: "models/yolox-s.onnx",
      classes_path: "models/coco_classes.json",
      model_impl: YOLO.Models.YOLOX
    )
    ```
  """
  @spec load(Keyword.t()) :: YOLO.Model.t()
  def load(options) do
    check_deprecated_default_model_impl(options)

    options = Keyword.merge(@default_load_options, options)
    model_impl = Keyword.fetch!(options, :model_impl)

    model_path = Keyword.fetch!(options, :model_path)
    classes_path = Keyword.fetch!(options, :classes_path)
    eps = Keyword.fetch!(options, :eps)
    json_decoder = Keyword.fetch!(options, :json_decoder)
    model_ref = Ortex.load(model_path, eps)
    classes = load_classes(classes_path, json_decoder)
    shapes = model_shapes(model_ref)

    Logger.info("Loaded model #{model_path} with #{inspect(eps)} execution providers")

    model = %YOLO.Model{
      ref: model_ref,
      classes: classes,
      model_impl: model_impl,
      shapes: shapes
    }

    model_impl.init(model, options)
  end

  defp check_deprecated_default_model_impl(options) do
    if not Keyword.has_key?(options, :model_impl) do
      Logger.warning("""
      DEPRECATION NOTICE: The default model implementation is now YOLO.Models.YOLOX.
      If you are using YOLO.Models.Ultralytics, please specify it explicitly with the :model_impl option.
      """)
    end
  end

  defp load_classes(classes_path, json_decoder) do
    classes_path
    |> File.read!()
    |> json_decoder.()
    |> Enum.with_index(fn class, idx -> {idx, class} end)
    |> Enum.into(%{})
  end

  @spec model_shapes(Ortex.Model.t()) :: %{(:input | :output) => tuple()}
  defp model_shapes(ref) do
    {[{_, _, input_shape}], [{_, _, output_shape}]} =
      Ortex.Native.show_session(ref.reference)

    %{input: List.to_tuple(input_shape), output: List.to_tuple(output_shape)}
  end

  @doc """
  Performs object detection on an image using a loaded YOLO model.

  ## Arguments
  * `model` - A loaded `YOLO.Model.t()` struct
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
  @spec detect(model :: YOLO.Model.t(), image :: term(), opts :: Keyword.t()) :: [[float()]]
  def detect(%{model_impl: model_impl} = model, image, opts \\ []) do
    opts = Keyword.merge(@default_detect_options, opts)

    {input_nx, scaling_config} = model_impl.preprocess(model, image, opts)
    output_nx = run(model, input_nx)
    model_impl.postprocess(model, output_nx, scaling_config, opts)
  end

  @doc """
  Runs inference on a preprocessed input tensor.

  ## Arguments
  * `model` - A loaded `YOLO.Model.t()` struct
  * `image_tensor` - Preprocessed input tensor matching model's expected shape

  ## Returns
  The raw output tensor from the model. For YOLOv8n:
  * Input shape: `{1, 3, 640, 640}`
  * Output shape: `{1, 84, 8400}`

  This is typically used internally by `detect/3` and shouldn't need to be called directly.
  """
  @spec run(YOLO.Model.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def run(model, image_tensor) do
    {output} = Ortex.run(model.ref, image_tensor)
    Nx.backend_transfer(output)
  end
end
