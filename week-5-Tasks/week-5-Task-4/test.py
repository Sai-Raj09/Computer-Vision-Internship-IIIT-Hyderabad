from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Predict on test images
results = model.predict(
    source="scaled_dataset/images/test",
    save=True,
    imgsz=384,
    conf=0.25
)

print("Prediction completed!")