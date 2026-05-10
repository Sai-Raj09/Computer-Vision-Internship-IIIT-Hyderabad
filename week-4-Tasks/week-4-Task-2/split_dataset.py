import os
import random
import shutil

source_folder = "frames"

train_folder = "yolo_dataset/images/train"
val_folder = "yolo_dataset/images/val"
test_folder = "yolo_dataset/images/test"

images = os.listdir(source_folder)

# Shuffle randomly
random.shuffle(images)

# Select images
train_images = images[:100]
val_images = images[100:140]
test_images = images[140:]

# Copy train images
for img in train_images:
    shutil.copy(os.path.join(source_folder, img), train_folder)

# Copy val images
for img in val_images:
    shutil.copy(os.path.join(source_folder, img), val_folder)

# Copy test images
for img in test_images:
    shutil.copy(os.path.join(source_folder, img), test_folder)

print("Dataset split completed successfully")