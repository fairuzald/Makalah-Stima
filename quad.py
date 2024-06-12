import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load Faster R-CNN model pre-trained on COCO dataset
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define transformation
transform = T.Compose([
    T.ToTensor()
])

# Function to apply the model on a patch
def detect_objects_on_patch(model, patch):
    patch_tensor = transform(patch).unsqueeze(0)
    with torch.no_grad():
        prediction = model(patch_tensor)
    return prediction

# Function to apply divide and conquer logic
def divide_and_conquer_detection(image, model, min_size=256, threshold=0.5):
    height, width = image.shape[:2]
    result_image = image.copy()
    total_objects = 0

    def divide_and_detect(x, y, w, h):
        nonlocal total_objects
        if w <= min_size or h <= min_size:
            patch = image[y:y+h, x:x+w]
            pil_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            predictions = detect_objects_on_patch(model, pil_patch)
            objects_in_patch = draw_predictions(result_image, predictions, x, y, threshold)
            total_objects += objects_in_patch
            return

        half_w = w // 2
        half_h = h // 2

        divide_and_detect(x, y, half_w, half_h)
        divide_and_detect(x + half_w, y, half_w, half_h)
        divide_and_detect(x, y + half_h, half_w, half_h)
        divide_and_detect(x + half_w, y + half_h, half_w, half_h)

    divide_and_detect(0, 0, width, height)
    return result_image, total_objects

# Function to draw predictions on the image
def draw_predictions(image, predictions, offset_x, offset_y, threshold):
    object_count = 0
    for element in zip(predictions[0]['boxes'], predictions[0]['scores']):
        box, score = element
        if score > threshold:
            x1, y1, x2, y2 = box.int().numpy()
            cv2.rectangle(image, (offset_x + x1, offset_y + y1), (offset_x + x2, offset_y + y2), (0, 255, 0), 2)
            object_count += 1
    return object_count

# Read the image
image = cv2.imread('8.jpg')

# Apply divide and conquer object detection
result_image, total_objects = divide_and_conquer_detection(image, model, min_size=256, threshold=0.5)

# Print the number of detected objects
print(f"Total number of objects detected: {total_objects}")

# Display the result
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Object Detection using Divide and Conquer with total objects detected: {}'.format(total_objects))
plt.axis('off')
plt.show()
