# Benchmarks

Below you find some Ultralytics `yolo11` benchmarks of **full pipeline** (preprocess, run, postprocess with NMS). 
You can run benchmarks yourself on your machine with

```bash
mix run benchmarks/ultralytics_yolo.exs <EPS> <MODEL_PATH>
```

* <EPS> is the mandatory first argument that should be one of: `cpu`, `coreml`, `directml`, `cuda`, or `tensorrt`. 
* <MODEL_PATH> is the required path to your ONNX file.

To set the proper Nx acceleration, configure your `config/config.exs` file with the appropriate backend.


* [Benchmarks on MacBook Air M3](benchmarks/macbook_air_m3.md)
