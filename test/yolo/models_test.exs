defmodule YOLO.ModelsTest do
  use ExUnit.Case

  @models_path Path.join(["test", "fixtures", "models"])

  describe "load/1" do
    test "if no classes_path is provided, it doesn't load classes" do
      model =
        YOLO.Models.load(
          model_path: Path.join(@models_path, "yolox_nano.onnx"),
          model_impl: YOLO.Models.YOLOX
        )

      assert model.classes == nil
    end
  end
end
