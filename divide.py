import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt
from PIL import Image

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to perform object detection on an image patch
def detect_objects(image, model, threshold=0.5):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

# Function to divide an image into patches
def divide_image(image, patch_size):
    patches = []
    h, w, _ = image.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append((patch, (x, y)))
    return patches

# Function to combine detection results from patches
def combine_detections(detections, patch_positions, patch_size, image_size):
    combined_boxes = []
    combined_labels = []
    combined_scores = []
    for (patch_detections, (x, y)) in zip(detections, patch_positions):
        boxes = patch_detections['boxes']
        labels = patch_detections['labels']
        scores = patch_detections['scores']
        for box, label, score in zip(boxes, labels, scores):
            if score >= 0.5:
                box[0] += x
                box[1] += y
                box[2] += x
                box[3] += y
                combined_boxes.append(box)
                combined_labels.append(label)
                combined_scores.append(score)
    return combined_boxes, combined_labels, combined_scores

# Load and preprocess the image
image = cv2.imread('8.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
patch_size = 500

# Divide the image into patches
patches = divide_image(image_rgb, patch_size)

# Detect objects in each patch
detections = []
for patch, position in patches:
    patch_image = Image.fromarray(patch)
    outputs = detect_objects(patch_image, model)
    detections.append(outputs[0])

# Combine detections from all patches
# Combine detections from all patches
combined_boxes, combined_labels, combined_scores = combine_detections(
    detections, [pos for _, pos in patches], patch_size, image_rgb.shape)

# Print the number of detected objects
print('Number of objects detected:', len(combined_boxes))

# Draw combined detection results on the original image
for box, label, score in zip(combined_boxes, combined_labels, combined_scores):
    cv2.rectangle(image_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.putText(image_rgb, f'{label.item()}: {score:.2f}', (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detection results
plt.imshow(image_rgb)
plt.title('Object Detection with Divide and Conquer')
plt.axis('off')
plt.show()
