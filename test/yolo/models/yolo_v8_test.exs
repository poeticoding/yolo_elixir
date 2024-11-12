defmodule YOLO.Models.YoloV8Test do
  use ExUnit.Case
  alias YOLO.Models.YoloV8

  import YOLO.Helpers

  @models_dir Path.join(["priv", "models"])
  @model_path Path.join(@models_dir, "yolov8n.onnx")
  @classes_path Path.join(@models_dir, "classes.json")

  @fixtures_path Path.join(["test", "fixtures"])
  # {640, 640, 3}
  @resized_image_path Path.join(@fixtures_path, "traffic640.jpg")
  # {900, 600, 3}
  @original_image_path Path.join(@fixtures_path, "traffic_original.jpg")

  setup_all _ctx do
    model = YOLO.Model.load(model_path: @model_path, classes_path: @classes_path, model_impl: YOLO.Models.YoloV8)
    resized_image = open_image_to_nx(@resized_image_path)
    original_image = open_image_to_nx(@original_image_path)

    # model output used to test postprocess/2
    model_output =
      resized_image
      |> YoloV8.preprocess()
      |> then(&YOLO.Model.run(model, &1))


    %{model: model, resized_image: resized_image, original_image: original_image, model_output: model_output}
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

  describe "run/2" do
    setup %{resized_image: image_nx} do
      # need to generate `input` for each test because
      # Ortex.Model.run/2 has side effects on the memory
      %{input: YoloV8.preprocess(image_nx)}
    end

    test "outputs a {1, 84, 8400} tensor", %{model: model, input: input} do
      assert {1, 84, 8400} = YOLO.Model.run(model, input).shape
    end
  end

  describe "postprocess/2" do
    test "returns the filtered result maps", %{model_output: model_output} do
      YoloV8.postprocess(model_output, [])
    end
  end
end
