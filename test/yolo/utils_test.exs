defmodule YOLO.UtilsTest do
  use ExUnit.Case, async: true

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
