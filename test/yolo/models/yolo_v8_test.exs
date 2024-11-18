defmodule YOLO.Models.YoloV8Test do
  use ExUnit.Case
  alias YOLO.Models.YoloV8

  import YOLO.Helpers

  # @classes_path Path.join(["priv", "models", "yolov8_coco_classes.json"])

  @fixtures_path Path.join(["test", "fixtures"])
  # {640, 640, 3}
  @resized_image_path Path.join(@fixtures_path, "traffic640.jpg")
  # {900, 600, 3}
  @original_image_path Path.join(@fixtures_path, "traffic_original.jpg")

  setup_all _ctx do
    resized_image = open_image_to_nx(@resized_image_path)
    original_image = open_image_to_nx(@original_image_path)

    # model output used to test postprocess/2
    model_output =
      @fixtures_path
      |> Path.join("traffic640_yolov8n_output.bin")
      |> File.read!()
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({84, 8400})

    %{resized_image: resized_image, original_image: original_image, model_output: model_output}
  end

  describe "preprocess/1" do
    test "returns a `{1, 3, 640, 640}`", %{resized_image: image_nx} do
      processed_image = YoloV8.preprocess(image_nx)
      assert {640, 640, 3} == image_nx.shape
      assert {1, 3, 640, 640} == processed_image.shape
    end

    test "raises an exception if the expected image shape is wrong" do
      assert_raise ArgumentError, fn ->
        YoloV8.preprocess(Nx.broadcast(0, {900, 640, 3}))
      end
    end
  end

  describe "postprocess/2" do
    test "returns the filtered result with probability >= 0.5", %{model_output: model_output} do
      detected_objects =
        YoloV8.postprocess(model_output, prob_threshold: 0.5, nms_iou_threshold: 0.5)

      assert Enum.count(detected_objects) > 0

      for [_cx, _cy, _w, _h, prob, class] <- detected_objects do
        assert 0.5 <= prob and prob <= 1

        # image is traffic640.jpg
        # 0 -> person
        # 2 -> car
        # 5 -> bus
        # 9 -> traffic light
        # 12 -> parking meter
        assert round(class) in [0, 2, 5, 9, 12]
      end
    end
  end
end
