defmodule YOLO.FrameScalers.ScalingConfig do
  @moduledoc """
  Stores image scaling and padding configuration used during YOLO model preprocessing.
  """

  @type t :: %__MODULE__{}

  defstruct original_image_shape: {640, 640},
            # scaled without padding
            scaled_image_shape: {640, 640},
            model_input_shape: {640, 640},
            # {width scale, height scale}
            scale: {1.0, 1.0},
            # {width padding, height padding}
            # 128 top and 128 bottom
            padding: {0, 0}
end
