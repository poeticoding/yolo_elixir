defmodule YOLO.Models do
  @moduledoc """
  This module handles loading YOLO models and running object detection on images.
  The `YOLO.Model` behaviour can be implemented for various YOLO variants.
  The supported models are:
  - `YOLO.Models.Ultralytics`: Implements models from the Ultralytics YOLO family (https://www.ultralytics.com).
  - `YOLO.Models.YOLOX`: Implements the YOLOX object detection model (https://github.com/Megvii-BaseDetection/YOLOX).


  ## Main Functions

  The key functions you'll use are:

  - `YOLO.Models.load/1`: Loads a YOLO model with required options
    ```elixir
    YOLO.Models.load(model_path: "path/to/model.onnx",
                    classes_path: "path/to/classes.json",
                    model_impl: YOLO.Models.Ultralytics)
    ```

  - `YOLO.Models.detect/3`: Runs object detection on an image
    ```elixir
    YOLO.Models.detect(model, image, prob_threshold: 0.5)
    ```
  """
  require Logger

  @default_load_options [
    model_impl: YOLO.Models.Ultralytics,
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

  ## Optional Options
  * `model_impl` - Module implementing the `YOLO.Model` behaviour (default: YOLO.Models.Ultralytics)
  * `classes_path` - Path to the `.json` file containing class labels. If not provided, classes will not be loaded.

  * `eps` - List of execution providers to pass to Ortex (e.g. `[:coreml]`, `[:cuda]`, `[:tensorrt]`, `[:directml]`), default: `[:cpu]`
  * `json_decoder` - Function to decode JSON strings (default: `&:json.decode/1`)

  ## Returns
  A `YOLO.Model.t()` struct containing:
  * `ref` - Reference to the loaded ONNX model
  * `model_impl` - The module implementing the model version
  * `classes` - Map of class indices to labels
  * `shapes` - Input/output tensor shapes
  * `model_data` - Model-specific data

  ## Examples
    ```elixir
    yolox_model = YOLO.Model.load(
      model_path: "models/yolox_s.onnx",
      classes_path: "models/coco_classes.json",
      model_impl: YOLO.Models.YOLOX
    )
    ```


    ```elixir
    ultralytics_model = YOLO.Model.load(
      model_path: "models/yolo11n.onnx",
      classes_path: "models/coco_classes.json",
      model_impl: YOLO.Models.Ultralytics
    )
    ```

  """
  @spec load(Keyword.t()) :: YOLO.Model.t()
  def load(options) do
    options = Keyword.merge(@default_load_options, options)
    model_impl = Keyword.fetch!(options, :model_impl)

    model_path = Keyword.fetch!(options, :model_path)
    classes_path = Keyword.get(options, :classes_path)
    eps = Keyword.fetch!(options, :eps)

    model_ref = Ortex.load(model_path, eps)
    Logger.info("Loaded model #{model_path} with #{inspect(eps)} execution providers")

    shapes = model_shapes(model_ref)

    model =
      %YOLO.Model{
        ref: model_ref,
        model_impl: model_impl,
        shapes: shapes
      }
      |> maybe_load_classes(classes_path, Keyword.take(options, [:json_decoder]))
      |> model_impl.init(options)

    Logger.info("Initialized model #{model_path}")
    model
  end

  @doc """
  Loads class labels from a JSON file and adds them to an existing model.

  ## Arguments
  * `model` - A `YOLO.Model.t()` struct to update with class labels
  * `classes_path` - Path to the JSON file containing class labels
  * `options` - Keyword list of options

  ## Options
  * `json_decoder` - Function to decode JSON strings (default: `&:json.decode/1`)

  ## Returns
  An updated `YOLO.Model.t()` struct with the `classes` field populated.
  """
  @spec load_classes(YOLO.Model.t(), String.t(), Keyword.t()) :: YOLO.Model.t()
  def load_classes(model, classes_path, options \\ []) do
    options = Keyword.validate!(options, json_decoder: @default_load_options[:json_decoder])
    json_decoder = options[:json_decoder]

    classes =
      classes_path
      |> File.read!()
      |> json_decoder.()
      |> Enum.with_index(fn class, idx -> {idx, class} end)
      |> Enum.into(%{})

    %{model | classes: classes}
  end

  defp maybe_load_classes(model, classes_path, options) do
    if classes_path do
      load_classes(model, classes_path, options)
    else
      model
    end
  end

  @spec model_shapes(term()) :: %{(:input | :output) => tuple()}
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
