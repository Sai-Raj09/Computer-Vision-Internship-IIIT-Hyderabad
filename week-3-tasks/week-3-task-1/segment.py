import os
from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolov8n-seg.pt")

input_folder = "images"
output_folder = "segmented"

os.makedirs(output_folder, exist_ok=True)

# Process all images
for img in sorted(os.listdir(input_folder)):
    img_path = os.path.join(input_folder, img)

    # Run segmentation
    results = model(img_path)

    # Save output
    for r in results:
        save_path = os.path.join(output_folder, img)
        r.save(filename=save_path)

print("Segmentation completed ✅")