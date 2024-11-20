defmodule YOLO.Models.YoloV8Test do
  use ExUnit.Case
  alias YOLO.Models.YoloV8

  @fixtures_path Path.join(["test", "fixtures"])

  setup_all _ctx do
    # model output used to test postprocess/2
    model_output =
      @fixtures_path
      |> Path.join("traffic_yolov8n_output.bin")
      |> File.read!()
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({84, 8400})

    %{model_output: model_output}
  end

  describe "preprocess/1" do
    test "returns a `{1, 3, 640, 640}`" do
      image_nx = Nx.iota({640, 640, 3})
      assert {1, 3, 640, 640} == YoloV8.preprocess(image_nx).shape
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
        YoloV8.postprocess(model_output, prob_threshold: 0.5, iou_threshold: 0.5)

      assert Enum.count(detected_objects) > 0

      for [_cx, _cy, _w, _h, prob, class] <- detected_objects do
        assert 0.5 <= prob and prob <= 1

        # image is traffic640.jpg
        # 0 -> person
        # 1 -> bicycle
        # 2 -> car
        # 5 -> bus
        # 7 -> truck
        # 9 -> traffic light
        # 12 -> parking meter
        assert round(class) in [0, 1, 2, 5, 7, 9, 12]
      end
    end
  end
end
