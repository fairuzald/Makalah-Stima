import cv2
import time

# Fungsi deteksi senyuman pada gambar penuh
def detect_smile_bruteforce(frame, face_cascade, smile_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            return True
    return False

# Fungsi deteksi senyuman dengan metode divide and conquer
def detect_smile_divide_and_conquer(frame, face_cascade, smile_cascade, grid_size=(2, 2)):
    height, width = frame.shape[:2]
    grid_h, grid_w = grid_size
    for i in range(grid_h):
        for j in range(grid_w):
            sub_img = frame[i * height // grid_h:(i + 1) * height // grid_h, j * width // grid_w:(j + 1) * width // grid_w]
            if detect_smile_bruteforce(sub_img, face_cascade, smile_cascade):
                return True
    return False

# Load haarcascades untuk deteksi wajah dan senyuman
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Brute Force Detection
    start_time = time.time()
    smile_detected_bruteforce = detect_smile_bruteforce(frame, face_cascade, smile_cascade)
    time_bruteforce = time.time() - start_time
    
    # Divide and Conquer Detection
    start_time = time.time()
    smile_detected_dnc = detect_smile_divide_and_conquer(frame, face_cascade, smile_cascade)
    time_dnc = time.time() - start_time
    
    # Display the results
    if smile_detected_bruteforce:
        cv2.putText(frame, "Smile Detected (Brute Force)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if smile_detected_dnc:
        cv2.putText(frame, "Smile Detected (Divide and Conquer)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display time taken for each method
    cv2.putText(frame, f"Time (Brute Force): {time_bruteforce:.2f} s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Time (Divide and Conquer): {time_dnc:.2f} s", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Smile Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
