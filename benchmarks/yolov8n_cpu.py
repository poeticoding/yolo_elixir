from ultralytics import YOLO
import time

model = YOLO("models/yolov8n.pt")

total = 0
count = 0
min_time = 10
max_time = -1

# warmup
for i in range(1, 10):
  model(["benchmarks/images/traffic.jpg"])

for i in range(1, 10):
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
