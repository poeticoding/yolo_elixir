defmodule YOLO.FrameScalersTest do
  use ExUnit.Case
  alias YOLO.FrameScalers
  alias YOLO.FrameScalers.ScalingConfig

  defmodule TestScaler do
    @behaviour YOLO.FrameScaler

    @impl true
    def image_shape(image) do
      # w, h, c
      image.shape
    end

    # returns {height, width, channels}
    @impl true
    def image_resize(_image, scaling_config) do
      {w, h} = scaling_config.scaled_image_shape
      Nx.broadcast(255, {h, w, 3})
    end

    # must return {height, width, channels}
    @impl true
    def image_to_nx(image) do
      # image is already a nx tensor
      # because `image_resize` returns a tensor of the given shape
      image
    end
  end

  describe "fit/3" do
    test "scales and pads image to target dimensions while preserving aspect ratio" do
      # Test with 1920x1080 input image scaling to 640x640
      # Note: target_shape is {width, height}
      image = Nx.broadcast(255, {1920, 1080, 3})
      {output_tensor, scaling_config} = FrameScalers.fit(image, {640, 640}, TestScaler)

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
      image = Nx.broadcast(255, {1920, 1080, 3})
      {output_tensor, _scaling_config} = FrameScalers.fit(image, {640, 640}, TestScaler)
      # Check padding area has value 114
      # Get a slice from the padded region (top padding area)
      # Shape is {height, width, channels}
      padding_slice = output_tensor[[0..139, 0..639, 0..2]]
      assert Nx.to_flat_list(padding_slice) |> Enum.all?(&(&1 == 114))
    end

    test "landscape image (width > height) scales correctly" do
      # original 1920x1080 -> Target: 640x640
      image = Nx.broadcast(255, {1920, 1080, 3})
      {output_tensor, scaling_config} = FrameScalers.fit(image, {640, 640}, TestScaler)

      # For landscape: width_ratio = 640/1920 = 0.333, height_ratio = 640/1080 = 0.593
      # width_ratio < height_ratio, so we should use width_ratio
      # scaled_width should be model_input_width (640)
      # scaled_height should be ceil(1080 * 0.333) = 360
      assert scaling_config.scaled_image_shape == {640, 360}
      # 140 padding on top and bottom
      assert scaling_config.padding == {0.0, 140.0}

      assert output_tensor.shape == {640, 640, 3}

      # output_tensor shape is HWC!
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[0..139, .., ..]], 114))) == 1

      # image
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[140..499, .., ..]], 255))) == 1

      # verify padding bottom
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[500..639, .., ..]], 114))) == 1
    end

    test "portrait image (height > width) scales correctly" do
      # Original: 1080x1920 -> Target: 640x640
      image = Nx.broadcast(255, {1080, 1920, 3})
      {output_tensor, scaling_config} = FrameScalers.fit(image, {640, 640}, TestScaler)

      # For portrait: width_ratio = 640/1080 = 0.593, height_ratio = 640/1920 = 0.333
      # height_ratio < width_ratio, so we should use height_ratio
      # scaled_width should be ceil(1080 * 0.333) = 360
      # scaled_height should be model_input_height (640)
      assert scaling_config.scaled_image_shape == {360, 640}
      # 140 padding on left and right
      assert scaling_config.padding == {140.0, 0.0}

      assert output_tensor.shape == {640, 640, 3}

      # output_tensor is HWC
      # verify padding top
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[.., 0..139, ..]], 114))) == 1

      # image
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[.., 140..499, ..]], 255))) == 1

      # verify padding bottom
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[.., 500..639, ..]], 114))) == 1
    end

    test "square image scales correctly" do
      # Target: 640x640
      image = Nx.broadcast(255, {1000, 1000, 3})

      {output_tensor, scaling_config} =
        FrameScalers.fit(image, {640, 640}, TestScaler)

      # For square: width_ratio = height_ratio = 640/1000 = 0.64
      # Since ratios are equal, we use the else branch (portrait logic)
      # scaled_width should be ceil(1000 * 0.64) = 640
      # scaled_height should be model_input_height (640)
      assert scaling_config.scaled_image_shape == {640, 640}
      assert scaling_config.padding == {0.0, 0.0}

      # image
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor, Nx.broadcast(255, {640, 640, 3})))) == 1
    end

    test "very wide landscape image scales correctly" do
      # Target: 640x640
      image = Nx.broadcast(255, {2560, 720, 3})

      {output_tensor, scaling_config} =
        FrameScalers.fit(image, {640, 640}, TestScaler)

      # For very wide: width_ratio = 640/2560 = 0.25, height_ratio = 640/720 = 0.889
      # width_ratio < height_ratio, so we should use width_ratio
      # scaled_width should be model_input_width (640)
      # scaled_height should be ceil(720 * 0.25) = 180
      assert scaling_config.scaled_image_shape == {640, 180}
      assert scaling_config.padding == {0.0, 230.0}

      # output_tensor shape is HWC!
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[0..229, .., ..]], 114))) == 1

      # image
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[230..409, .., ..]], 255))) == 1

      # verify padding bottom
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[411..639, .., ..]], 114))) == 1
    end

    test "very tall portrait image scales correctly" do
      # Target: 640x640
      image = Nx.broadcast(255, {720, 2560, 3})

      {output_tensor, scaling_config} =
        FrameScalers.fit(image, {640, 640}, TestScaler)

      # For very tall: width_ratio = 640/720 = 0.889, height_ratio = 640/2560 = 0.25
      # height_ratio < width_ratio, so we should use height_ratio
      # scaled_width should be ceil(720 * 0.25) = 180
      # scaled_height should be model_input_height (640)
      assert scaling_config.scaled_image_shape == {180, 640}
      assert scaling_config.padding == {230.0, 0.0}

      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[.., 0..229, ..]], 114))) == 1

      # image
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[.., 230..409, ..]], 255))) == 1

      # verify padding bottom
      assert Nx.to_number(Nx.all(Nx.equal(output_tensor[[.., 410..639, ..]], 114))) == 1
    end
  end
end
