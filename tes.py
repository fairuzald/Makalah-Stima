import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
from    torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and prepare the image
image_path = '8.jpg'
image = Image.open(image_path).convert("RGB")

# Convert image to tensor
image_tensor = F.to_tensor(image)

# Function to perform object detection on an image tensor
def detect_objects(model, image_tensor):
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    return predictions

# Function to draw bounding boxes on the image
def draw_boxes(image, predictions, threshold=0.5):
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Function to resize image tensor
def resize_image_tensor(image_tensor, scale):
    _, h, w = image_tensor.shape
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = F.resize(image_tensor, [new_h, new_w])
    return resized_image

# Apply Decrease and Conquer strategy
scales = [1.0, 0.75, 0.5, 0.25]
fig, axs = plt.subplots(1, len(scales), figsize=(20, 5))

max_objects = 0
max_objects_scale = None

for i, scale in enumerate(scales):
    # Resize image
    resized_image_tensor = resize_image_tensor(image_tensor, scale)
    
    # Perform object detection
    predictions = detect_objects(model, resized_image_tensor)
    
    # Filter predictions by threshold
    scores = predictions['scores'].cpu().numpy()
    num_objects = (scores >= 0.5).sum()
    
    # Track the scale with the maximum number of objects
    if num_objects > max_objects:
        max_objects = num_objects
        max_objects_scale = scale
    
    # Convert tensor to numpy array for visualization
    resized_image_np = resized_image_tensor.permute(1, 2, 0).cpu().numpy()
    resized_image_np = (resized_image_np * 255).astype(np.uint8)
    
    # Convert to BGR for OpenCV
    resized_image_np = cv2.cvtColor(resized_image_np, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes
    result_image = draw_boxes(resized_image_np, predictions)
    
    # Convert back to RGB for display
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Display the result
    axs[i].imshow(result_image)
    axs[i].set_title(f'Scale: {scale}')
    axs[i].axis('off')

plt.show()

print(f'Scale with maximum objects: {max_objects_scale}, Number of objects: {max_objects}')
