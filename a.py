import cv2
import numpy as np
import os

# Load YOLOv3 model
def load_yolo():
    net = cv2.dnn.readNet("yolo7.weights", "yolo7.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Function to detect objects
def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return class_ids, confidences, boxes

# Function to remove noise
def remove_noise(img, class_ids, confidences, boxes, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if classes[class_ids[i]] == 'noise_class':  # Replace 'noise_class' with the class you consider as noise
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return img

# Divide and Conquer Approach
def divide_and_conquer(img, net, output_layers, classes, patch_size=1000):
    height, width, _ = img.shape
    result_img = np.zeros_like(img)
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            class_ids, confidences, boxes = detect_objects(patch, net, output_layers)
            patch = remove_noise(patch, class_ids, confidences, boxes, classes)
            result_img[y:y+patch_size, x:x+patch_size] = patch
    
    return result_img

# Conventional Approach
def conventional_approach(img, net, output_layers, classes):
    class_ids, confidences, boxes = detect_objects(img, net, output_layers)
    result_img = remove_noise(img, class_ids, confidences, boxes, classes)
    return result_img

# Load image
image_path = "nois.jpg"
img = cv2.imread(image_path)

# Load YOLO
net, classes, output_layers = load_yolo()

# Apply Divide and Conquer
result_divide_and_conquer = divide_and_conquer(img.copy(), net, output_layers, classes)

# Apply Conventional Approach
result_conventional = conventional_approach(img.copy(), net, output_layers, classes)

# Save results
cv2.imwrite("result_divide_and_conquer.jpg", result_divide_and_conquer)
cv2.imwrite("result_conventional.jpg", result_conventional)

# Display results
cv2.imshow("Divide and Conquer", result_divide_and_conquer)
cv2.imshow("Conventional", result_conventional)
cv2.waitKey(0)
cv2.destroyAllWindows()
