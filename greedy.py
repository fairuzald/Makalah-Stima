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

# Function to perform object detection on an image
def detect_objects(image, model, threshold=0.5):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

# Function to compute IoU (Intersection over Union) between two boxes
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union

# Function to perform greedy optimization on detections
def greedy_optimization(boxes, scores, threshold=0.5, iou_threshold=0.5):
    sorted_indices = np.argsort(-scores)
    selected_boxes = []
    selected_scores = []

    for i in sorted_indices:
        box = boxes[i]
        score = scores[i]
        if score < threshold:
            break
        if all(compute_iou(box, selected_box) < iou_threshold for selected_box in selected_boxes):
            selected_boxes.append(box)
            selected_scores.append(score)
    
    return selected_boxes, selected_scores

# Load and preprocess the image
image = cv2.imread('8.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

# Perform object detection
outputs = detect_objects(image_pil, model)

# Extract boxes and scores
boxes = outputs[0]['boxes']
scores = outputs[0]['scores']

# Apply greedy optimization
optimized_boxes, optimized_scores = greedy_optimization(boxes, scores)

# Draw bounding boxes on the image
for box, score in zip(optimized_boxes, optimized_scores):
    box = [int(coord) for coord in box]
    cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(image_rgb, f'Score: {score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Print the number of detected objects
object_count = len(optimized_boxes)
print(f'Number of objects detected: {object_count}')

# Display the image with bounding boxes
plt.imshow(image_rgb)
plt.title(f'Object Detection with Greedy Optimization ({object_count} objects detected)')
plt.axis('off')
plt.show()
