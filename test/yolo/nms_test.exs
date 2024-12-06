defmodule YOLO.NMSTest do
  use ExUnit.Case
  alias YOLO.NMS

  @fixtures_path Path.join(["test", "fixtures"])
  setup_all _ctx do
    nms_input =
      @fixtures_path
      |> Path.join("traffic_yolov8n_output.bin")
      |> File.read!()
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({84, 8400})
      |> Nx.transpose(axes: [1, 0])

    %{input: nms_input}
  end

  describe "nms/2" do
    test "when all rows have same bbox, same prob, same class, discards all the bboxes but first" do
      bbox = [100.0, 100.0, 30.0, 30.0]
      # prob > 0.5
      # class = 0
      input = [
        bbox ++ [0.81, 0.0],
        bbox ++ [0.8, 0.0],
        bbox ++ [0.6, 0.0]
      ]

      # iou > 0.5 -> discard
      # keeping only the first row
      assert [bbox ++ [0.81, 0.0]] == NMS.nms(input, 0.5)
    end

    test "nms is run separately for each class" do
      bbox = [100.0, 100.0, 30.0, 30.0]
      # prob > 0.5
      # class = 0
      people = [
        bbox ++ [0.81, 0.0],
        bbox ++ [0.8, 0.0],
        bbox ++ [0.6, 0.0]
      ]

      # same bbox as people
      bikes = [
        bbox ++ [0.81, 1.0],
        bbox ++ [0.8, 1.0],
        bbox ++ [0.6, 1.0]
      ]

      # keeping one bbox for person and one bbox for bike
      assert MapSet.new([
               # person
               bbox ++ [0.81, 0.0],
               # bike
               bbox ++ [0.81, 1.0]
             ]) == MapSet.new(NMS.nms(people ++ bikes, 0.5))
    end
  end

  describe "filter_predictions/2" do
    test "filters out rows with max detection prob under 0.7", %{input: input} do
      assert {8400, 84} == input.shape

      filtered_probs =
        input
        |> NMS.filter_predictions(0.7)
        |> Enum.map(fn [_, _, _, _, prob, _class] -> prob end)

      assert Enum.count(filtered_probs) > 0
      for p <- filtered_probs, do: assert(p >= 0.7)
    end
  end

  describe "iou/2" do
    test "small iou" do
      # [cx, cy, w, h]
      a = [3.5, 3, 7, 6]
      b = [9, 7, 8, 6]

      # intersection = 4
      # union = 86
      assert NMS.iou(a, b) == 4 / 86
    end

    test "iou > 0.5. Two bboxes with the same center with same area" do
      a = [4, 3.5, 6, 8]
      b = [4, 3.5, 8, 6]

      # intersection = 6*6 = 36
      # union = 6*8 + 8*6 - 36 = 60
      # iou = 36/60 = 0.6
      assert NMS.iou(a, b) == 36 / 60
    end

    test "iou = 1 when bboxes are equal" do
      assert NMS.iou([10, 10, 5, 5], [10, 10, 5, 5]) == 1
    end

    test "iou = 1 when bboxes have intersection = 0" do
      assert NMS.iou([10, 10, 5, 5], [100, 100, 5, 5]) == 0
    end
  end
end
