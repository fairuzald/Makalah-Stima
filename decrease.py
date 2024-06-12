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

# Function to decrease and conquer
def decrease_and_conquer(image, model, initial_scale=0.5, steps=3, threshold=0.5):
    h, w, _ = image.shape
    current_scale = initial_scale
    detections = None
    
    for step in range(steps):
        # Resize image to current scale
        resized_image = cv2.resize(image, (int(w * current_scale), int(h * current_scale)))
        image_pil = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        
        # Perform object detection
        outputs = detect_objects(image_pil, model)
        
        if detections is None:
            detections = outputs[0]
        else:
            # Scale up the bounding boxes from the previous detection
            scale_factor = 1 / current_scale
            for i, box in enumerate(detections['boxes']):
                detections['boxes'][i] = box * scale_factor
            
            # Merge the new detections with the previous ones
            new_boxes = outputs[0]['boxes'] * scale_factor
            new_scores = outputs[0]['scores']
            detections['boxes'] = torch.cat((detections['boxes'], new_boxes))
            detections['scores'] = torch.cat((detections['scores'], new_scores))
        
        # Increase the scale for the next iteration
        current_scale = min(1.0, current_scale * 2)
    
    # Filter out the final detections based on the threshold
    final_boxes = []
    final_scores = []
    for box, score in zip(detections['boxes'], detections['scores']):
        if score >= threshold:
            final_boxes.append(box)
            final_scores.append(score)
    
    return final_boxes, final_scores

# Load and preprocess the image
image = cv2.imread('6.jpg')

# Perform decrease and conquer detection
boxes, scores = decrease_and_conquer(image, model)

# Draw bounding boxes on the image
for box, score in zip(boxes, scores):
    box = [int(coord) for coord in box]
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(image, f'Score: {score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Print the number of detected objects
object_count = len(boxes)
print(f'Number of objects detected: {object_count}')

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Object Detection with Decrease and Conquer ({object_count} objects detected)')
plt.axis('off')
plt.show()
