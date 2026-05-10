from ultralytics import YOLO
#Load your trained model
model=YOLO("yolov8n-seg.pt")
#Validate your model
metrics = model.val(data="coco128-seg.yaml")
print(metrics)