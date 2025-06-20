defmodule YOLO.Model do
  @moduledoc """
  Defines a behaviour for implementing YOLO object detection models.

  This module provides the structure for loading and running YOLO models for object detection.
  The default implementation is `YOLO.Models.YoloV8`, but you can create custom implementations
  for other YOLO variants.

  ## Required Callbacks

  To implement this behaviour, you need to define these functions:

  - `preprocess/3`: Prepares an input image for the model
    - Takes a model struct, input image, and options
    - Returns `{preprocessed_tensor, scaling_config}`
    - See `YOLO.Models.YoloV8` for an example implementation

  - `postprocess/4`: Processes the model's raw output into detected objects
    - Takes model struct, model output tensor, scaling config, and options
    - Returns list of detected objects as `[cx, cy, w, h, prob, class_idx]`
    - Handles tasks like non-maximum suppression and coordinate scaling

  ## Optional Callbacks

  - `init/1`: Initializes the model. This is called when the model is loaded. It's the place where you can, for example, generate and store in `model_data` things like grids and expanded strides that will be used later in other callbacks.

  ## Types

  - `t()`: The model struct containing:
    - `:ref` - Reference to loaded ONNX model
    - `:model_impl` - Module implementing this behaviour
    - `:shapes` - Input/output tensor shapes
    - `:classes` - Map of class indices to labels
    - `:model_data` - Model-specific data, e.g. grids and expanded strides for YOLOX

  - `detected_object()`: Map containing detection results:
    - `:bbox` - Bounding box coordinates (cx, cy, w, h)
    - `:class` - Detected class name
    - `:class_idx` - Class index
    - `:prob` - Detection probability
  """
  alias YOLO.FrameScalers.ScalingConfig

  @enforce_keys [:ref, :model_impl, :shapes]
  defstruct [:ref, :classes, :model_impl, :shapes, :model_data]

  @type classes :: %{integer() => String.t()}

  @type t :: %__MODULE__{
          ref: term(),
          shapes: %{(:input | :output) => tuple()},
          # module implementing the behaviour
          model_impl: module(),
          classes: classes() | nil,
          model_data: term() | nil
        }

  @type shape :: {integer(), integer()}

  @type detected_object :: %{
          # Object bounding box. cx, cy, w, h
          bbox: %{cx: integer(), cy: integer(), w: integer(), h: integer()},
          # object class name
          class: String.t(),
          # class index
          class_idx: integer(),
          # detection probability
          prob: float()
        }

  @doc """
  Initialize the model. This is called when the model is loaded. It's the place where you can, for example, generate and store in `model_data` things like grids and expanded strides that will be used later in other callbacks.

  ## Parameters
    * `model`
    * `options` : options passed to YOLO.load/2

  ## Returns
    * The updated `YOLO.Model` struct
  """
  @callback init(model :: t(), options :: Keyword.t()) :: t()

  @doc """
  Prepares input image tensors for the model.

  ## Parameters
    * `model` - The YOLO.Model struct containing model information
    * `image` - Input image in implementation's native format (e.g. Evision.Mat)
    * `options` - Keyword list of options:
      * `:frame_scaler` - Module implementing YOLO.FrameScaler behaviour (required)

  ## Returns
    * `{input_tensor, scaling_config}` tuple where:
      * `input_tensor` is the preprocessed Nx tensor ready for model input, where shape is `{1, channels, height, width}`
      * `scaling_config` contains scaling/padding info for postprocessing

  Look at the `YOLO.Models.YoloV8.preprocess/3` implementation to see how this callback is implemented.

  """
  @callback preprocess(model :: t(), image :: term(), options :: Keyword.t()) ::
              {Nx.Tensor.t(), ScalingConfig.t()}

  @doc """
  Post-processes the model's raw output to produce a list of detected objects.

  The raw output from the model is a tensor containing bounding box coordinates and class probabilities
  for each candidate detection.

  For example, YOLOv8 outputs a `{1, 84, 8400}` tensor where:
    - 84 represents 4 bbox coordinates + 80 class probabilities
    - 8400 represents the number of candidate detections

  Returns a list of detections where each detection is a list of 6 elements:
  ```elixir
  [cx, cy, w, h, prob, class_idx]
  ```
  where:
  - `cx`, `cy`: center x,y coordinates of bounding box
  - `w`, `h`: width and height of bounding box
  - `prob`: detection probability
  - `class_idx`: class index

  The implementation should:
  1. Filter low probability detections
  2. Apply non-maximum suppression (NMS) to remove overlapping boxes
  3. Scale back the coordinates using the `scaling_config` and `YOLO.FrameScaler` module, since
     the detections are based on the model's input resolution rather than the original image size

  See `YOLO.Models.YoloV8.postprocess/4` for a reference implementation.
  """
  @callback postprocess(
              model :: t(),
              model_output :: Nx.Tensor.t(),
              scaling_config :: ScalingConfig.t(),
              options :: Keyword.t()
            ) :: [
              [float()]
            ]
end
