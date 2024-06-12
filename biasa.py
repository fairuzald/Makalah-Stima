import cv2
from fer import FER

def detect_emotion(frame):
    # Initialize the FER detector
    detector = FER(mtcnn=True)
    
    # Detect emotions
    result = detector.detect_emotions(frame)
    
    return result

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect emotion
    result = detect_emotion(frame)
    
    # Display the results
    if result:
        emotions = result[0]['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        emotion_text = f"Emotion: {dominant_emotion}"
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
