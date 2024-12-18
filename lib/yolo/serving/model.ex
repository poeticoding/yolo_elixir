defmodule Yolo.Serving.Model do
  @moduledoc """
  GenServer to maintain Model state
  """

  use GenServer

  @enforce_keys [:ref, :name, :model_impl, :shapes]
  @fields [:pid, :name, :ref, :classes, :model_impl, :shapes]

  @type classes :: %{integer() => String.t()}

  @type t :: %__MODULE__{
          pid: pid(),
          name: module(),
          ref: term(),
          shapes: %{(:input | :output) => tuple()},
          model_impl: module(),
          classes: classes()
        }

  @derive {Jason.Encoder, only: @fields}

  defstruct @fields

  # API
  @spec start_link(any()) :: :ignore | {:error, any()} | {:ok, pid()}
  def start_link(model) do
    GenServer.start_link(__MODULE__, model, name: get_pid(model.name))
  end

  @spec get_model(any) :: any
  def get_model(model_name) do
    GenServer.call(get_pid(model_name), :get_model)
  end

  def terminate(model_name) do
    GenServer.call(get_pid(model_name), :terminate)
  end

  # GENSERVER
  def init(model) do
    {:ok,
     %__MODULE__{
       pid: get_pid(model.name),
       name: model.name,
       ref: model.ref,
       classes: model.classes,
       model_impl: model.model_impl,
       shapes: model.shapes
     }}
  end

  def handle_call(:get_model, _from, state) do
    {:reply, state, state}
  end

  def handle_call(:terminate, _from, state) do
    {:stop, :normal, :ok, state}
  end

  defp get_pid(model_name) do
    {:via, Registry, {:model_registry, model_name}}
  end
end
