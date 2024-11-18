defmodule YOLO.Helpers do
  @moduledoc false
  def open_image_to_nx(path) do
    path
    |> Image.open!()
    |> Image.to_nx!()
  end
end
