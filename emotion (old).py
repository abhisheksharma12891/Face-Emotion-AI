import cv2
from deepface import DeepFace

# 1. Turn on the webcam (0 means the laptop's default camera)
cap = cv2.VideoCapture(0)

# 2. Load the face recognition tool (it comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🎥 Camera starting... Please wait (First time load takes time!)")

while True:
    # Draw a frame (photo)
    ret, frame = cap.read()
    if not ret:
        break

    # Convert photo to grayscale (Black & White) (faster for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw a box around the face (Blue Color)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        try:
            # --- SMART WORK (AI MAGIC) ---
            # Cut out only the face portion
            face_img = frame[y:y+h, x:x+w]
            
            # Ask DeepFace: "What emotion is on this face?"
            # actions=['emotion'] means we only need the emotion, not the age/gender
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            
            # Pick out the most important emotions (e.g. 'happy', 'sad', 'angry')
            emotion = result[0]['dominant_emotion']

            # Write Emotion on Screen (Green Color Text)
            cv2.putText(frame, emotion.upper(), (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        except Exception as e:
            pass # If the face is blurred, ignore it.

    # Show video on screen
    cv2.imshow('Abhishek AI Face Scanner', frame)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanliness (Stop littering)
cap.release()
cv2.destroyAllWindows()