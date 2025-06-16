## v0.2.0 (2025-06)

### Enhancements
  * Added YOLOX support - thanks @aspett (Andrew Pett)!
  * Added `init/2` callback to the `YOLO.Model` behaviour for model initialization
  * Model-agnostic postprocessing:
    * Removed fixed `{8400, 84}` shape constraint
    * Now supports dynamic shapes like `{batch_size, num_detections, bbox_coords + num_classes}`
    * Enables use of custom-trained models with varying class counts
  * Optimized NMS implementation:
    * Rewritten using `Nx.Defn` for 100x performance improvement
    * Added support for variable detection counts and class numbers
  * Improved Ultralytics preprocessing:
    * New implementation using `defn`:
      * 1.18K iterations/second, 846.54µs per operation, 19.67KB memory
    * Old implementation:
      * 458 iterations/second, 2.18ms per operation, 44.59KB memory
  * Made `:classes_path` optional in `YOLO.load/1`

### Refactoring
* `yolox_nano.onnx`, `yolox_s.onnx`, `coco_classes.json`  available under `models/` directory.
* 
### Deprecations
* fully removed `YOLO.Models.Yolov8` in favor of `YOLO.Models.Ultralytics`.


## v0.1.2 (2024-12-30)

### Enhancements
* Added `:eps` option to `YOLO.Models.load/1` for specifying execution providers to pass to Ortex.

### Refactorings 
* Moved `YOLO.Models.YoloV8` logic to `YOLO.Models.Ultralytics`.

### Deprecations
* Deprecated `YOLO.Models.YoloV8`.

### Chores
* Updated `Ortex` to `0.1.10`.