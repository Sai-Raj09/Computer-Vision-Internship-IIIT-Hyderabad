from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("yolov8n.pt")

input_folder = "images"
output_folder = "detect_images"

os.makedirs(output_folder, exist_ok=True)

image_files = sorted(os.listdir(input_folder))

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)

    frame = cv2.imread(img_path)
    if frame is None:
        continue

    # Run detection
    results = model(frame)
    annotated = results[0].plot()

    # Save output
    out_path = os.path.join(output_folder, img_name)
    cv2.imwrite(out_path, annotated)

print("✅ Detection images saved!")