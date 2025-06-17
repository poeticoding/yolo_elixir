from ultralytics import YOLO
import sys
import time

model_path = sys.argv[1]
# default to cpu, can be "mps" or "cuda"
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

model = YOLO(model_path)

total = 0
count = 0
min_time = 10
max_time = -1

# warmup
for i in range(1, 10):
  model(["benchmarks/images/traffic.jpg"], device=device)
  # Mac
  # model(["benchmarks/images/traffic.jpg"], device='mps')
  
  # CUDA
  # model(["benchmarks/images/traffic.jpg"], device='cuda')

for i in range(0, 10):
  start = time.time()
  model(["benchmarks/images/traffic.jpg"])
  stop = time.time()
  diff = stop - start
  count += 1
  total += diff
  if min_time > diff:
    min_time = diff
  if max_time < diff:
    max_time = diff

print("count: ", count)
print("avg: ", total/count)
print("min: ", min_time)
print("max: ", max_time)
