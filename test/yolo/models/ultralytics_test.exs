defmodule YOLO.Models.UltralyticsTest do
  use ExUnit.Case
  alias YOLO.Models.Ultralytics

  @fixtures_path Path.join(["test", "fixtures"])

  setup_all _ctx do
    model = %YOLO.Model{
      ref: nil,
      shapes: %{
        input: {1, 3, 640, 640},
        output: {1, 84, 8400}
      },
      model_impl: Ultralytics
    }

    # model output used to test postprocess/2
    model_output =
      @fixtures_path
      |> Path.join("traffic_yolov8n_output.bin")
      |> File.read!()
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({84, 8400})

    %{model: model, model_output: model_output}
  end

  describe "preprocess/3" do
    defmodule TestScaler do
      @behaviour YOLO.FrameScaler

      @impl true
      def image_shape(_image) do
        {1920, 1080, 3}
      end

      @impl true
      def image_resize(_image, _scaling_config) do
        # Return dummy image data
        :test_image
      end

      @impl true
      def image_to_nx(_image) do
        # Return tensor with shape matching scaled dimensions
        # Note: image_shape returns {height, width, channels}
        Nx.broadcast(1, {360, 640, 3})
      end
    end

    test "returns a `{1, 3, 640, 640}` tensor and scaling config", %{model: model} do
      image_nx = Nx.iota({640, 640, 3})

      {output_nx, scaling_config} =
        Ultralytics.preprocess(model, image_nx, frame_scaler: TestScaler)

      assert {1, 3, 640, 640} == Nx.shape(output_nx)
      assert %YOLO.FrameScalers.ScalingConfig{} = scaling_config
    end
  end

  describe "postprocess/4" do
    test "returns the filtered result with probability >= 0.5", %{
      model: model,
      model_output: model_output
    } do
      scaling_config = %YOLO.FrameScalers.ScalingConfig{
        original_image_shape: {1920, 1080},
        scaled_image_shape: {640, 360},
        model_input_shape: {640, 640},
        scale: {0.3333333333333333, 0.3333333333333333},
        padding: {0.0, 140.0}
      }

      detected_objects =
        Ultralytics.postprocess(model, model_output, scaling_config,
          prob_threshold: 0.25,
          iou_threshold: 0.45
        )

      assert Enum.count(detected_objects) > 0

      for [_cx, _cy, _w, _h, prob, _class] <- detected_objects do
        assert 0.25 <= prob and prob <= 1
      end

      # Count occurrences of each class
      class_counts =
        detected_objects
        |> Enum.map(fn [_, _, _, _, _, class] -> trunc(class) end)
        |> Enum.frequencies()

      assert class_counts[0] == 18
      assert class_counts[1] == 3
      assert class_counts[2] == 6
      assert class_counts[7] == 1
      assert class_counts[9] == 2
    end
  end
end
