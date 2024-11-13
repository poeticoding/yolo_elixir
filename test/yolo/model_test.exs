# describe "run/2" do
#   setup %{resized_image: image_nx} do
#     # need to generate `input` for each test because
#     # Ortex.Model.run/2 has side effects on the memory
#     %{input: YoloV8.preprocess(image_nx)}
#   end

#   test "outputs a {1, 84, 8400} tensor", %{model: model, input: input} do
#     assert {1, 84, 8400} = YOLO.Model.run(model, input).shape
#   end
# end
