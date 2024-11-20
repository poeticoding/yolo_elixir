defmodule YOLO.UtilsTest do
  use ExUnit.Case, async: true

  describe "scale_bboxes_to_original/3" do
    test "rescales the bbox coordinates and shape" do
      # w_ratio = 900/640 = 1.40625
      # h_ratio = 600/640 = 0.9375
      #
      # cx * w_ratio = 140.625 -> round -> 141
      # cy *  h_ratio = 93.75 -> round -> 94
      # w * w_ratio = 28.125 -> round -> 28
      # h * h_ratio = 18.75 -> round -> 19

      assert [[141, 94, 28, 19, 0.5, 2]] =
               YOLO.Utils.scale_bboxes_to_original(
                 [
                   # cx, cy, w, h, prob, class
                   [100, 100, 20, 20, 0.5, 2]
                 ],
                 {640, 640},
                 {900, 600}
               )
    end
  end

  test "to_detected_objects/2" do
    classes = %{0 => "person", 1 => "bicycle", 2 => "car"}

    assert [
             %{bbox: %{cx: 100, cy: 100, w: 5, h: 10}, prob: 0.7, class: "person", class_idx: 0},
             %{bbox: %{cx: 20, cy: 50, w: 20, h: 20}, prob: 0.5, class: "car", class_idx: 2}
           ] ==
             YOLO.Utils.to_detected_objects(
               [
                 [100, 100, 5, 10, 0.7, 0],
                 [20, 50, 20, 20, 0.5, 2]
               ],
               classes
             )
  end
end
