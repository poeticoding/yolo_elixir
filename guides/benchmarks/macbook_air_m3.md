# Benchmarks on MacBook Air M3

### Elixir YOLO

EPS: `coreml` 

**Ultralytics YOLO v11 nano** - `yolo11n`

```
> mix run benchmarks/ultralytics_yolo.exs coreml models/yolo11n.onnx

Operating System: macOS
CPU Information: Apple M3
Number of Available Cores: 8
Available memory: 16 GB
Elixir 1.18.3
Erlang 27.3.3
JIT enabled: true
```

| Name | ips | average  | median | 99th %
| ---  | --- | ---      | ---    |
| preprocess | 975.38 | 1.03 ms | 0.99 ms | 1.38 ms
| postprocess | 276.86 | 3.61 ms |3.56 ms | 3.98 ms
| run         | 64.07 | 15.61 ms | 15.63 ms | 17.66 ms

Around 20ms from the frame to the detections, which means ~50 FPS, not bad!

**Ultralytics YOLO v11 medium** - `yolo11m`

```terminal
> mix run benchmarks/ultralytics_yolo.exs coreml models/yolo11m.onnx

Operating System: macOS
CPU Information: Apple M3
Number of Available Cores: 8
Available memory: 16 GB
Elixir 1.18.3
Erlang 27.3.3
JIT enabled: true
```

| Name | ips | average  | median | 99th %
| ---  | --- | ---      | ---    |
| preprocess | 892.35 | 1.12 ms | 1.09 ms | 1.67 ms
| postprocess | 265.45 | 3.77 ms | 3.70 ms | 4.36 ms
| run | 25.20  | 39.68 ms  | 39.68 ms       43.92 ms


Around 22FPS.

### Python Ultralytics
Let's see how performs, on the same machine, what I consider the benchmark - the official Ultralytics library:

**Ultralytics YOLO v11 nano** - `yolo11n`

```terminal
> python benchmarks/ultralytics_yolo.py models/yolo11n.pt cpu

0: 384x640 17 persons, 4 bicycles, 7 cars, 1 truck, 1 traffic light, 25.8ms
Speed: 1.2ms preprocess, 25.8ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
```

Pre and post-processing are faster (not by much) and inference is definitely slower than the elixir version, probably due to Ortex coreml acceleration.

When running the benchmark with `mps` acceleration (if you have a Mac with Apple Silicon), inference is much faster (even than Elixir YOLO with ONNX model), but pre and post-processing are slower.

```
> python benchmarks/ultralytics_yolo.py models/yolo11n.pt mps

0: 384x640 17 persons, 4 bicycles, 7 cars, 1 truck, 1 traffic light, 9.2ms
Speed: 1.4ms preprocess, 9.2ms inference, 5.3ms postprocess per image at shape (1, 3, 384, 640)
```

**Ultralytics YOLO v11 medium** - `yolo11m`


```terminal
> python benchmarks/ultralytics_yolo.py models/yolo11m.pt mps

0: 384x640 16 persons, 7 bicycles, 10 cars, 1 truck, 4 traffic lights, 3 backpacks, 1 handbag, 23.3ms
Speed: 1.5ms preprocess, 23.3ms inference, 11.8ms postprocess per image at shape (1, 3, 384, 640)
```

