defmodule YOLO.Serving.ModelTest do
  use ExUnit.Case

  describe "start_link/2" do
    test "when create a new model for serving" do
      Registry.start_link(keys: :unique, name: :model_registry)

      %{
        name: __MODULE__,
        ref: nil,
        shapes: %{
          input: {1, 3, 640, 640},
          output: {1, 84, 8400}
        },
        model_impl: YOLO.Models.YoloV8,
        classes: %{}
      }
      |> Yolo.Serving.Model.start_link()

      assert %Yolo.Serving.Model{
               pid: {:via, Registry, {:model_registry, YOLO.Serving.ModelTest}},
               name: YOLO.Serving.ModelTest,
               ref: nil,
               classes: %{},
               model_impl: YOLO.Models.YoloV8,
               shapes: %{input: {1, 3, 640, 640}, output: {1, 84, 8400}}
             } = Yolo.Serving.Model.get_model(__MODULE__)
    end
  end
end
