defmodule YOLO.Integration.YOLOXTest do
  @moduledoc """
  `test/fixtures/yolox_*.onnx` models are downloaded from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime)
  """
  use ExUnit.Case

  @models_path "models/"
  @classes_path Path.join(@models_path, "coco_classes.json")
  @yolox_nano_path Path.join(@models_path, "yolox_nano.onnx")
  @yolox_s_path Path.join(@models_path, "yolox_s.onnx")

  @fixtures_path "test/fixtures"
  @dog_image_path Path.join(@fixtures_path, "dog.jpg")

  setup_all do
    yolox_nano =
      YOLO.load(
        model_path: @yolox_nano_path,
        classes_path: @classes_path,
        model_impl: YOLO.Models.YOLOX
      )

    yolox_s =
      YOLO.load(
        model_path: @yolox_s_path,
        classes_path: @classes_path,
        model_impl: YOLO.Models.YOLOX
      )

    %{yolox_nano: yolox_nano, yolox_s: yolox_s}
  end

  describe "yolox_nano" do
    setup %{yolox_nano: model} do
      %{model: model}
    end

    test "dog.jpg", %{model: model} do
      mat = Evision.imread(@dog_image_path)

      detected_objects =
        model
        # decode_head default is `true`, but making it evident
        |> YOLO.detect(mat, decode_head: true, prob_threshold: 0.25, iou_threshold: 0.5)
        |> YOLO.to_detected_objects(model.classes)

      ## PROB > 0.25
      assert %{"bicycle" => 2, "dog" => 1, "truck" => 1, "car" => 1} ==
               classes_count(detected_objects)

      detected_objects =
        model
        |> YOLO.detect(mat, decode_head: true, prob_threshold: 0.5, iou_threshold: 0.5)
        |> YOLO.to_detected_objects(model.classes)

      ## PROB > 0.5
      assert %{"bicycle" => 1, "dog" => 1, "truck" => 1} == classes_count(detected_objects)
    end
  end

  describe "yolox_s" do
    setup %{yolox_s: model} do
      %{model: model}
    end

    test "dog.jpg", %{model: model} do
      mat = Evision.imread(@dog_image_path)

      detected_objects =
        model
        # decode_head default is `true`, but making it evident
        |> YOLO.detect(mat, decode_head: true, prob_threshold: 0.25, iou_threshold: 0.5)
        |> YOLO.to_detected_objects(model.classes)

      ## PROB > 0.25
      assert %{"bicycle" => 1, "dog" => 1, "truck" => 1, "car" => 1, "potted plant" => 1} ==
               classes_count(detected_objects)

      detected_objects =
        model
        |> YOLO.detect(mat, decode_head: true, prob_threshold: 0.5, iou_threshold: 0.5)
        |> YOLO.to_detected_objects(model.classes)

      ## PROB > 0.5
      assert %{"bicycle" => 1, "dog" => 1, "truck" => 1, "car" => 1} ==
               classes_count(detected_objects)
    end
  end

  defp classes_count(detections) do
    detections
    |> Enum.group_by(& &1.class)
    |> Enum.reduce(%{}, fn {k, v}, acc ->
      Map.put(acc, k, Enum.count(v))
    end)
  end
end
