defmodule YOLO.FrameScalersTest do
  use ExUnit.Case
  alias YOLO.FrameScalers
  alias YOLO.FrameScalers.ScalingConfig

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

  describe "fit/3" do
    test "scales and pads image to target dimensions while preserving aspect ratio" do
      # Test with 1920x1080 input image scaling to 640x640
      # Note: target_shape is {width, height}
      {output_tensor, scaling_config} = FrameScalers.fit(:test_image, {640, 640}, TestScaler)

      assert scaling_config == %ScalingConfig{
               # Note: shape stored as {width, height}
               original_image_shape: {1920, 1080},
               scaled_image_shape: {640, 360},
               model_input_shape: {640, 640},
               scale: {0.3333333333333333, 0.3333333333333333},
               padding: {0.0, 140.0}
             }

      # Output tensor shape is {height, width, channels}
      assert Nx.shape(output_tensor) == {640, 640, 3}
    end

    test "pads with gray value 114" do
      {output_tensor, _scaling_config} = FrameScalers.fit(:test_image, {640, 640}, TestScaler)
      # Check padding area has value 114
      # Get a slice from the padded region (top padding area)
      # Shape is {height, width, channels}
      padding_slice = output_tensor[[0..139, 0..639, 0..2]]
      assert Nx.to_flat_list(padding_slice) |> Enum.all?(&(&1 == 114))
    end
  end
end
