defmodule YOLO.FrameScalers.EvisionScalerTest do
  use ExUnit.Case
  alias YOLO.FrameScalers.EvisionScaler

  @traffic_image_path "test/fixtures/traffic.jpg"

  setup do
    mat = Evision.imread(@traffic_image_path)
    {:ok, mat: mat}
  end

  describe "image_shape/1" do
    test "returns width, height and channels of evision mat", %{mat: mat} do
      assert EvisionScaler.image_shape(mat) == {1920, 1080, 3}
    end
  end

  describe "image_resize/2" do
    test "resizes image according to scaling config", %{mat: mat} do
      assert mat.shape == {1080, 1920, 3}

      scaling_config = %YOLO.FrameScalers.ScalingConfig{
        scaled_image_shape: {640, 360}
      }

      resized = EvisionScaler.image_resize(mat, scaling_config)

      {height, width, channels} = Evision.Mat.shape(resized)

      assert width == 640
      assert height == 360
      assert channels == 3
    end
  end

  describe "image_to_nx/1" do
    test "converts evision mat to nx tensor with correct shape", %{mat: mat} do
      tensor = EvisionScaler.image_to_nx(mat)

      assert Nx.shape(tensor) == {1080, 1920, 3}
      assert Nx.type(tensor) == {:u, 8}
    end
  end
end
