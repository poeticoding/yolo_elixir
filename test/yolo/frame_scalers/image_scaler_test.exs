defmodule YOLO.FrameScalers.ImageScalerTest do
  use ExUnit.Case
  alias YOLO.FrameScalers.ImageScaler

  @traffic_image_path "test/fixtures/traffic.jpg"

  setup do
    image = Image.open!(@traffic_image_path)
    {:ok, image: image}
  end

  describe "image_shape/1" do
    test "returns width, height and channels of vix image", %{image: image} do
      assert ImageScaler.image_shape(image) == {1920, 1080, 3}
    end
  end

  describe "image_resize/2" do
    test "resizes image according to scaling config", %{image: image} do
      {width, height, channels} = Image.shape(image)
      assert width == 1920
      assert height == 1080
      assert channels == 3

      scaling_config = %YOLO.FrameScalers.ScalingConfig{
        scaled_image_shape: {640, 360},
        scale: {0.3333333333333333, 0.3333333333333333}
      }

      resized = ImageScaler.image_resize(image, scaling_config)

      {width, height, channels} = Image.shape(resized)

      assert width == 640
      assert height == 360
      assert channels == 3
    end
  end

  describe "image_to_nx/1" do
    test "converts vix image to nx tensor with correct shape", %{image: image} do
      tensor = ImageScaler.image_to_nx(image)

      # {height, width, channels}
      assert Nx.shape(tensor) == {1080, 1920, 3}
      assert Nx.type(tensor) == {:u, 8}
    end
  end
end
