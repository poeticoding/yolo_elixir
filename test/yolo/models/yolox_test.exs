defmodule YOLO.Models.YOLOXTest do
  @moduledoc """
  At the moment, `postprocess/4` is indirectly covered in the integration tests
  """
  use ExUnit.Case
  alias YOLO.Models.YOLOX
  alias YOLO.FrameScalers.ScalingConfig

  describe "init/2" do
    setup do
      model_8400 = %YOLO.Model{
        ref: nil,
        shapes: %{input: {1, 3, 640, 640}, output: {1, 8400, 84}},
        model_impl: YOLOX
      }

      model_3549 = %YOLO.Model{
        ref: nil,
        shapes: %{input: {1, 3, 416, 416}, output: {1, 3549, 84}},
        model_impl: YOLOX
      }

      %{model_8400: model_8400, model_3549: model_3549}
    end

    # 52x52 + 26x26 + 13x13 grids = 3549
    test "grids for nano and tiny models (416x416 input and 3549 output detections)", %{
      model_3549: model
    } do
      %{model_data: %{grids: grids}} = YOLOX.init(model, [])
      assert grids.shape == {1, 3549, 2}
      # X coordinates
      # 52x52 grid
      offset = 0
      assert equal?(grids[[0, 0..51, 0]], Nx.tensor(Enum.to_list(0..51)))
      # 26x26 grid
      offset = offset + 52 * 52
      assert equal?(grids[[0, offset..(offset + 25), 0]], Nx.tensor(Enum.to_list(0..25)))
      # 13x13 grid
      offset = offset + 26 * 26
      assert equal?(grids[[0, offset..(offset + 12), 0]], Nx.tensor(Enum.to_list(0..12)))

      # Y coordinates
      # 52x52 grid
      offset = 0
      assert equal?(grids[[0, 0..51, 1]], Nx.broadcast(0, {52}))
      offset = offset + 52 * 52
      assert Nx.to_number(grids[[0, offset - 1, 1]]) == 51

      # 26x26 grid
      assert equal?(grids[[0, offset..(offset + 25), 1]], Nx.broadcast(0, {26}))
      offset = offset + 26 * 26
      assert Nx.to_number(grids[[0, offset - 1, 1]]) == 25

      # 13x13 grid
      assert equal?(grids[[0, offset..(offset + 12), 1]], Nx.broadcast(0, {13}))
      offset = offset + 13 * 13
      assert Nx.to_number(grids[[0, offset - 1, 1]]) == 12
    end

    # 80x80 + 40x40 + 20x20 grids = 8400
    test "grids for s, m, l, x (640x640 input and 8400 output detections)", %{model_8400: model} do
      %{model_data: %{grids: grids}} = YOLOX.init(model, [])
      assert grids.shape == {1, 8400, 2}
      # X coordinates
      # 80x80 grid
      offset = 0
      assert equal?(grids[[0, 0..79, 0]], Nx.tensor(Enum.to_list(0..79)))
      # 40x40 grid
      offset = offset + 80 * 80
      assert equal?(grids[[0, offset..(offset + 39), 0]], Nx.tensor(Enum.to_list(0..39)))
      # 20x20 grid
      offset = offset + 40 * 40
      assert equal?(grids[[0, offset..(offset + 19), 0]], Nx.tensor(Enum.to_list(0..19)))

      # Y coordinates
      # 80x80 grid
      offset = 0
      assert equal?(grids[[0, 0..79, 1]], Nx.broadcast(0, {80}))
      offset = offset + 80 * 80
      assert Nx.to_number(grids[[0, offset - 1, 1]]) == 79

      # 40x40 grid
      assert equal?(grids[[0, offset..(offset + 39), 1]], Nx.broadcast(0, {40}))
      offset = offset + 40 * 40
      assert Nx.to_number(grids[[0, offset - 1, 1]]) == 39

      # 20x20 grid
      assert equal?(grids[[0, offset..(offset + 19), 1]], Nx.broadcast(0, {20}))
      offset = offset + 20 * 20
      assert Nx.to_number(grids[[0, offset - 1, 1]]) == 19
    end

    test "strides for 416x416 and 3549 output", %{model_3549: model} do
      %{model_data: %{expanded_strides: expanded_strides}} = YOLOX.init(model, [])
      # 416/52 = 8
      # 416/26 = 16
      # 416/13 = 32
      assert equal?(expanded_strides[[0, 0..(52 * 52 - 1), 0]], Nx.broadcast(8, {52 * 52}))
      offset = 52 * 52

      assert equal?(
               expanded_strides[[0, offset..(offset + 26 * 26 - 1), 0]],
               Nx.broadcast(16, {26 * 26})
             )

      offset = offset + 26 * 26

      assert equal?(
               expanded_strides[[0, offset..(offset + 13 * 13 - 1), 0]],
               Nx.broadcast(32, {13 * 13})
             )
    end

    test "strides for 640x640 and 8400 output", %{model_8400: model} do
      %{model_data: %{expanded_strides: expanded_strides}} = YOLOX.init(model, [])
      # 640/80 = 8
      # 640/40 = 16
      # 640/20 = 32
      assert equal?(expanded_strides[[0, 0..(80 * 80 - 1), 0]], Nx.broadcast(8, {80 * 80}))
      offset = 80 * 80

      assert equal?(
               expanded_strides[[0, offset..(offset + 40 * 40 - 1), 0]],
               Nx.broadcast(16, {40 * 40})
             )

      offset = offset + 40 * 40

      assert equal?(
               expanded_strides[[0, offset..(offset + 20 * 20 - 1), 0]],
               Nx.broadcast(32, {20 * 20})
             )
    end
  end

  describe "preprocess/3" do
    setup do
      model = %YOLO.Model{
        ref: nil,
        shapes: %{
          input: {1, 3, 640, 640},
          output: {1, 85, 8400}
        },
        model_impl: YOLOX
      }

      %{model: model}
    end

    defmodule TestScaler do
      @behaviour YOLO.FrameScaler

      @impl true
      def image_shape(_image) do
        {640, 640, 3}
      end

      @impl true
      def image_resize(_image, _scaling_config) do
        # Return dummy image data
        :test_image
      end

      @impl true
      def image_to_nx(_image) do
        # 640x640x3 and value 200 1-255
        Nx.broadcast(200, {640, 640, 3})
      end
    end

    test "returns a `{1, 3, 640, 640}` tensor and scaling config", %{model: model} do
      image_nx = Nx.iota({640, 640, 3})

      {output_nx, scaling_config} =
        YOLOX.preprocess(model, image_nx, frame_scaler: TestScaler)

      assert {1, 3, 640, 640} == Nx.shape(output_nx)
      assert %ScalingConfig{} = scaling_config
    end

    test "does not normalize input", %{model: model} do
      image_nx = Nx.broadcast(200, {640, 640, 3})

      {output_nx, _scaling_config} =
        YOLOX.preprocess(model, image_nx, frame_scaler: TestScaler)

      # just transposed
      assert equal?(output_nx, Nx.broadcast(200, {1, 3, 640, 640}) |> Nx.as_type({:f, 32}))
    end
  end

  describe "postprocess/4 - decode head 3549x85 model output" do
    setup do
      model =
        YOLOX.init(
          %YOLO.Model{
            ref: nil,
            shapes: %{input: {1, 3, 416, 416}, output: {1, 3549, 85}},
            model_impl: YOLOX
          },
          []
        )

      scaling_config = %ScalingConfig{
        original_image_shape: {416, 416},
        scaled_image_shape: {416, 416},
        model_input_shape: {416, 416},
        scale: {1.0, 1.0},
        padding: {0, 0}
      }

      %{model: model, scaling_config: scaling_config}
    end

    test "bboxes detected in 52x52 grid", %{model: model, scaling_config: scaling_config} do
      # stride = 416/52 = 8px
      # center of the cell 0.5, 0.5 and h = 1.5, w = 0.
      # width = exp(w) * stride
      # width = exp(0) * 8 = 8px
      # height = exp(1.5) * 8 = 40px
      # center of the image 26th row and 26th column
      # 26x26 cell
      detection_idx = 52 * 26 + 26

      model_output =
        Nx.broadcast(0, {1, 3549, 85})
        # first 4 values are the bbox coordinates and size
        # 5th value is 100% of object presence confidence
        # 6th 1.0 is 100% confidence on person
        |> Nx.put_slice([0, detection_idx, 0], Nx.tensor([[[0.5, 0.5, 0.0, 1.5, 0.8, 0.8]]]))

      # 0.8 * 0.8 = 0.64 objectness for that cell * person class confidence
      assert [[212, 212, w, h, score, +0.0]] =
               YOLOX.postprocess(model, model_output, scaling_config,
                 decode_head: true,
                 prob_threshold: 0.5,
                 iou_threshold: 0.5
               )

      assert round(w) == 8
      assert round(h) == 36
      assert round(score * 100) == 64
    end

    test "bboxes detected in 26x26 grid", %{model: model, scaling_config: scaling_config} do
      # stride = 416/26 = 16px
      # width = exp(0) * 16 = 16px
      # height = exp(1.5) * 16 = 72px

      detection_idx = 52 * 52 + 26 * 13 + 13

      model_output =
        Nx.broadcast(0, {1, 3549, 85})
        # 0.7 car
        |> Nx.put_slice(
          [0, detection_idx, 0],
          Nx.tensor([[[0.5, 0.5, 0.0, 1.5, 0.8, 0.0, 0.0, 0.7]]])
        )

      # 0.8 * 0.7 = 0.56 objectness for that cell * car class confidence
      assert [[216, 216, w, h, score, 2.0]] =
               YOLOX.postprocess(model, model_output, scaling_config,
                 decode_head: true,
                 prob_threshold: 0.5,
                 iou_threshold: 0.5
               )

      assert round(w) == 16
      assert round(h) == 72
      assert round(score * 100) == 56
    end

    test "bboxes detected in 13x13 grid", %{model: model, scaling_config: scaling_config} do
      # stride = 416/13 = 32px
      # width = exp(0) * 32 = 32px
      # height = exp(1.5) * 32 = 143px

      detection_idx = 52 * 52 + 26 * 26 + 13 * 7 + 6

      model_output =
        Nx.broadcast(0, {1, 3549, 85})
        |> Nx.put_slice(
          [0, detection_idx, 0],
          Nx.tensor([[[0.5, 0.25, 0.0, 1.5, 0.8, 0.0, 0.8]]])
        )

      # x = 6*32 + 32*0.5 = 208
      # y = 7*32 + 32*0.25 = 232
      # 0.8 * 0.8 = 0.64 objectness for that cell * bicycle class confidence
      assert [[208, 232, w, h, score, 1.0]] =
               YOLOX.postprocess(model, model_output, scaling_config,
                 decode_head: true,
                 prob_threshold: 0.5,
                 iou_threshold: 0.5
               )

      assert round(w) == 32
      assert round(h) == 143
      assert round(score * 100) == 64
    end
  end

  describe "postprocess/4 - decode head 8400x85 model output" do
    setup do
      model =
        YOLOX.init(
          %YOLO.Model{
            ref: nil,
            shapes: %{input: {1, 3, 640, 640}, output: {1, 8400, 85}},
            model_impl: YOLOX
          },
          []
        )

      scaling_config = %ScalingConfig{
        original_image_shape: {640, 640},
        scaled_image_shape: {640, 640},
        model_input_shape: {640, 640},
        scale: {1.0, 1.0},
        padding: {0, 0}
      }

      %{model: model, scaling_config: scaling_config}
    end

    test "bboxes detected in 80x80 grid", %{model: model, scaling_config: scaling_config} do
      # stride = 640/80 = 8px
      detection_idx = 80 * 40 + 40

      model_output =
        Nx.broadcast(0, {1, 8400, 85})
        |> Nx.put_slice([0, detection_idx, 0], Nx.tensor([[[0.5, 0.5, 0.0, 1.5, 0.8, 0.8]]]))

      assert [[324, 324, w, h, score, +0.0]] =
               YOLOX.postprocess(model, model_output, scaling_config,
                 decode_head: true,
                 prob_threshold: 0.5,
                 iou_threshold: 0.5
               )

      assert round(w) == 8
      assert round(h) == 36
      assert round(score * 100) == 64
    end

    test "bboxes detected in 40x40 grid", %{model: model, scaling_config: scaling_config} do
      # stride = 640/40 = 16px
      detection_idx = 80 * 80 + 40 * 20 + 20

      model_output =
        Nx.broadcast(0, {1, 8400, 85})
        |> Nx.put_slice(
          [0, detection_idx, 0],
          Nx.tensor([[[0.5, 0.5, 0.0, 1.5, 0.8, 0.8]]])
        )

      assert [[328, 328, w, h, score, +0.0]] =
               YOLOX.postprocess(model, model_output, scaling_config,
                 decode_head: true,
                 prob_threshold: 0.5,
                 iou_threshold: 0.5
               )

      assert round(w) == 16
      assert round(h) == 72
      assert round(score * 100) == 64
    end

    test "bboxes detected in 20x20 grid", %{model: model, scaling_config: scaling_config} do
      # stride = 640/20 = 32px
      detection_idx = 80 * 80 + 40 * 40 + 20 * 10 + 10

      model_output =
        Nx.broadcast(0, {1, 8400, 85})
        |> Nx.put_slice(
          [0, detection_idx, 0],
          Nx.tensor([[[0.5, 0.25, 0.0, 1.5, 0.8, 0.8]]])
        )

      assert [[336, 328, w, h, score, +0.0]] =
               YOLOX.postprocess(model, model_output, scaling_config,
                 decode_head: true,
                 prob_threshold: 0.5,
                 iou_threshold: 0.5
               )

      assert round(w) == 32
      assert round(h) == 143
      assert round(score * 100) == 64
    end
  end

  defp equal?(a, b) do
    Nx.equal(a, b) |> Nx.all() |> Nx.to_number() == 1
  end
end
