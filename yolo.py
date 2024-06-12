import os
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        try:
            # Initialize YOLO model
            self.net = cv2.dnn.readNet("yolo7.weights", "yolo7.cfg")
            self.classes = []
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            raise e

    def detect_objects(self, img, use_divide_conquer=False, tile_size=100, confidence_threshold=0.2):
        try:
            # Preprocess the image
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            output_layers = self.net.getUnconnectedOutLayersNames()
            outputs = self.net.forward(output_layers)

            # Initialize detected objects list
            detected_objects = []

            # Process detections
            height, width = img.shape[:2]
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Filter detections by confidence threshold
                    if confidence > confidence_threshold:
                        # Calculate bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Add detected object to the list
                        detected_objects.append((x, y, w, h, self.classes[class_id], confidence))

            # Apply divide and conquer if specified
            if use_divide_conquer:
                detected_objects = self.divide_and_conquer(img, detected_objects, tile_size, confidence_threshold)

            return detected_objects
        except Exception as e:
            raise e

    def divide_and_conquer(self, img, detected_objects, tile_size, confidence_threshold):
        try:
            height, width = img.shape[:2]
            new_objects = []

            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    tile = img[y:y+tile_size, x:x+tile_size]
                    tile_objects = self.detect_objects(tile, confidence_threshold=confidence_threshold)

                    for obj in tile_objects:
                        x_off, y_off, w, h, class_name, confidence = obj
                        x_abs = x + x_off
                        y_abs = y + y_off
                        new_objects.append((x_abs, y_abs, w, h, class_name, confidence))

            return new_objects
        except Exception as e:
            raise e

# Load image
image_path = "s/8.jpg"
image = cv2.imread(image_path)

# Initialize object detector
detector = ObjectDetector()

# Detect objects using normal approach
detected_objects_normal = detector.detect_objects(image)

# Detect objects using divide and conquer approach
detected_objects_dc = detector.detect_objects(image, use_divide_conquer=True, tile_size=200, confidence_threshold=0.2)

# Draw bounding boxes on the normal detection image
image_normal = image.copy()
for obj in detected_objects_normal:
    x, y, w, h, class_name, confidence = obj
    color = (0, 255, 0)  # Green color for bounding box
    cv2.rectangle(image_normal, (x, y), (x + w, y + h), color, 2)
    text = f"{class_name}: {confidence:.2f}"
    cv2.putText(image_normal, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Draw bounding boxes on the divide and conquer detection image
image_dc = image.copy()
for obj in detected_objects_dc:
    x, y, w, h, class_name, confidence = obj
    color = (0, 255, 0)  # Green color for bounding box
    cv2.rectangle(image_dc, (x, y), (x + w, y + h), color, 2)
    text = f"{class_name}: {confidence:.2f}"
    cv2.putText(image_dc, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display both images
cv2.imshow("Normal Detection", image_normal)
cv2.imshow("Divide and Conquer Detection", image_dc)
print(len(detected_objects_normal), len(detected_objects_dc))
cv2.waitKey(0)
cv2.destroyAllWindows()
