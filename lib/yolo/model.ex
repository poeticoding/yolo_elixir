defmodule YOLO.Model do
  @enforce_keys [:ref, :model_impl]
  defstruct [:ref, :classes, :model_impl]

  @type t :: %__MODULE__{
          ref: Ortex.Model.t(),
          # module implementing the behaviour
          model_impl: module()
        }

  @type detected_object :: %{
          # Object bounding box. cx, cy, w, h
          bbox: {integer(), integer(), integer(), integer()},
          # object class name
          class: String.t(),
          # detection probability
          prob: float()
        }

  @default_load_options [model_impl: YOLO.Models.YoloV8]
  @default_detect_options [prob_threshold: 0.5, nms_iou_threshold: 0.5]

  @callback preprocess(Nx.Tensor.t()) :: Nx.Tensor.t()
  @callback postprocess(Nx.Tensor.t(), Keyword.t()) :: [detected_object()]

  @doc """
  Loads the model and returns a `YOLO.Model.t()` struct where:
  * `ref`: the ortex model (at the moment just using Ortex to load onnx models).
  * `model_impl`: the module implementing the model version.
  * `classes`: list of object class.

  `options`:
  * `model_path`: path to the `.onnx` you want to load. (Required)
  * `classes_path`: path to the `.json` file containing the YOLO classes. (Required)
  * `model_impl`: the module implementing the `YOLO.Model` behaviour. (Default is `YOLO.Models.YoloV8`).

  """
  @spec load(Keyword.t()) :: t()
  def load(options) do
    options = Keyword.merge(@default_load_options, options)
    model_impl = Keyword.fetch!(options, :model_impl)
    model_path = Keyword.fetch!(options, :model_path)
    classes_path = Keyword.fetch!(options, :classes_path)
    model = Ortex.load(model_path)
    classes = File.read!(classes_path)
    %__MODULE__{ref: model, classes: classes, model_impl: model_impl}
  end

  @doc """
  Returns the detect object found in the given image.

  """
  @spec detect(t(), Nx.Tensor.t(), Keyword.t()) :: [Model.detected_object()]
  def detect(%{model_impl: impl} = model, image_nx, opts \\ []) do
    opts = Keyword.merge(@default_detect_options, opts)
    image_nx = impl.preprocess(image_nx)
    run(model, image_nx)
    impl.postprocess(opts)
  end

  @doc """
  Runs the model for the given tensor and returns the output tensor.
  * `model` is `YOLO.Model.t()`.
  * `image_tensor` shape.

  YoloV8n example:
  * `image_tensor` shape is `{1, 3, 640, 640}`
  * returned tensor shape is `{1, 84, 8400}`
  """
  @spec run(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def run(model, image_tensor) do
    {output} = Ortex.run(model.ref, image_tensor)
    Nx.backend_transfer(output)
  end
end
