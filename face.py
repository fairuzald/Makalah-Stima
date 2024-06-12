import cv2
import time
from fer import FER
import numpy as np
import glob

# Fungsi deteksi emosi tanpa divide and conquer
def detect_emotion(frame):
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(frame)
    if result:
        return result[0]['emotions']
    return None

# Fungsi deteksi emosi dengan divide and conquer
def divide_and_conquer_detect_emotion(frame, grid_size=(2, 2)):
    height, width, _ = frame.shape
    detector = FER(mtcnn=True)
    grid_h, grid_w = grid_size
    sub_images = []
    for i in range(grid_h):
        for j in range(grid_w):
            sub_img = frame[i * height // grid_h:(i + 1) * height // grid_h, j * width // grid_w:(j + 1) * width // grid_w]
            sub_images.append(sub_img)
    results = []
    for sub_img in sub_images:
        result = detector.detect_emotions(sub_img)
        if result:
            results.append(result[0]['emotions'])
    if results:
        average_emotions = {}
        for emotion in results[0].keys():
            average_emotions[emotion] = np.mean([r[emotion] for r in results])
        return average_emotions
    return None

# Fungsi untuk menguji akurasi dan waktu eksekusi
def test_accuracy_and_time(image_paths, labels):
    accuracy_without_dc = 0
    accuracy_with_dc = 0
    time_without_dc = 0
    time_with_dc = 0
    
    if not image_paths:
        print("No images found.")
        return
    
    for image_path, label in zip(image_paths, labels):
        frame = cv2.imread(image_path)
        
        start_time = time.time()
        emotions_without_dc = detect_emotion(frame)
        end_time = time.time()
        if emotions_without_dc:
            dominant_emotion_without_dc = max(emotions_without_dc, key=emotions_without_dc.get)
            if dominant_emotion_without_dc == label:
                accuracy_without_dc += 1
        time_without_dc += end_time - start_time
        
        start_time = time.time()
        emotions_with_dc = divide_and_conquer_detect_emotion(frame)
        end_time = time.time()
        if emotions_with_dc:
            dominant_emotion_with_dc = max(emotions_with_dc, key=emotions_with_dc.get)
            if dominant_emotion_with_dc == label:
                accuracy_with_dc += 1
        time_with_dc += end_time - start_time
    
    total_images = len(image_paths)
    accuracy_without_dc = (accuracy_without_dc / total_images) * 100
    accuracy_with_dc = (accuracy_with_dc / total_images) * 100
    avg_time_without_dc = time_without_dc / total_images
    avg_time_with_dc = time_with_dc / total_images
    
    return accuracy_without_dc, accuracy_with_dc, avg_time_without_dc, avg_time_with_dc
image_paths = glob.glob("images/train/happy/*.jpg")

# Limit to 10 images
image_paths = image_paths[:10]
labels = ['happy'] * len(image_paths)  # Assuming all images are labeled as 'happy'

# Validate that image paths are not empty
if not image_paths:
    print("No images found in the specified directory.")
else:
    # Uji akurasi dan waktu eksekusi
    accuracy_without_dc, accuracy_with_dc, avg_time_without_dc, avg_time_with_dc = test_accuracy_and_time(image_paths, labels)

    print(f"Accuracy without divide and conquer: {accuracy_without_dc}%")
    print(f"Accuracy with divide and conquer: {accuracy_with_dc}%")
    print(f"Average time without divide and conquer: {avg_time_without_dc} seconds")
    print(f"Average time with divide and conquer: {avg_time_with_dc} seconds")
