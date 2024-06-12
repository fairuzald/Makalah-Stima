import torch
import torchvision
import cv2
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

# Load and preprocess the image
image = cv2.imread('9.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

# Perform object detection
outputs = detect_objects(image_pil, model)

# Draw bounding boxes on the image
boxes = outputs[0]['boxes']
labels = outputs[0]['labels']
scores = outputs[0]['scores']

object_count = 0
for box, label, score in zip(boxes, labels, scores):
    if score >= 0.5:
        object_count += 1
        box = [int(coord) for coord in box]
        cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image_rgb, f'{label.item()}: {score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(image_rgb)
plt.title(f'Object Detection with Faster R-CNN ({object_count} objects detected)')
plt.axis('off')
plt.show()
