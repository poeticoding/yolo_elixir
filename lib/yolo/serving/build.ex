defmodule Yolo.Serving.Build do
  @moduledoc """
  Serving encapsulates client and server work to perform batched requests.
  """

  alias Yolo.Serving.Model

  @doc """
  Function to create a Nx.Serving and struct model.

  ## Parameters
    * `serving_name` - Module that defines Nx.Serving. (e.g. MyServing)
    * `model_path` - path/to/model.onnx
    * `classes_path` - path/to/classes.json
    * `options` - Keyword list of options

  Before creating a new model, you need to define it in the Application:
    {Registry, [keys: :unique, name: :model_registry]}

  Used to store the pid of models created in Nx.Serving


  Now, you can create a several new Nx.Serving/Model
    Yolo.Serving.Build.call(
      MyServing,
      Path.join(__DIR__, "models/yolov8n.onnx"),
      Path.join(__DIR__, "models/yolov8n_classes.json"),
      []
    )

    Yolo.Serving.Build.call(
      MyOtherServing,
      Path.join(__DIR__, "models/other_yolo.onnx"),
      Path.join(__DIR__, "models/other_yolo_classes.json"),
      []
    )

  For each serving will be create a %Yolo.Serving.Model{}

  ## Types
  - `t()`: The model struct containing:
    - `:name` - Serving name
    - `:model_ref` - Reference to loaded ONNX model
    - `:model_impl` - Module implementing this behaviour
    - `:shapes` - Input/output tensor shapes
    - `:classes` - Map of class indices to labels

  Returns a Nx.Serving to predict objects
  """
  @spec call(module(), binary(), binary(), keyword()) :: term()
  def call(serving_name, model_path, classes_path, options) do
    batch_size = Keyword.get(options, :batch_size, 10)
    batch_timeout = Keyword.get(options, :batch_timeout, 100)
    partitions = Keyword.get(options, :partitions, true)
    model_ref = Ortex.load(model_path)

    {:ok, _pid} =
      %{
        name: serving_name,
        ref: model_ref,
        shapes: model_shapes(model_ref),
        model_impl: YOLO.Models.YoloV8,
        classes: load_classes(classes_path)
      }
      |> Model.start_link()

    {
      Nx.Serving,
      serving: build_serving(model_ref),
      name: serving_name,
      batch_size: batch_size,
      batch_timeout: batch_timeout,
      partitions: partitions
    }
  end

  defp build_serving(model_ref) do
    Ortex.Serving
    |> Nx.Serving.new(model_ref, defn_options: [compiler: Torchx.Backend, client: :mps])
    |> Nx.Serving.client_preprocessing(fn input -> {input, :any_data} end)
  end

  defp model_shapes(ref) do
    {[{_, _, input_shape}], [{_, _, output_shape}]} =
      Ortex.Native.show_session(ref.reference)

    %{input: List.to_tuple(input_shape), output: List.to_tuple(output_shape)}
  end

  defp load_classes(classes_path) do
    classes_path
    |> File.read!()
    |> :json.decode()
    |> Enum.with_index(fn class, idx -> {idx, class} end)
    |> Enum.into(%{})
  end
end
