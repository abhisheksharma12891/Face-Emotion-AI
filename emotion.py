import cv2
from deepface import DeepFace

# --- 1. SETUP & CONFIGURATION ---

# Music Recommendation Logic (The "Smart Twist" for Interview)
music_dict = {
    'happy': 'Happy by Pharrell Williams 🎵',
    'sad': 'Someone Like You by Adele 🎻',
    'angry': 'Believer by Imagine Dragons 🔥',
    'neutral': 'Lo-Fi Beats ☕',
    'surprise': 'Wow by Post Malone 😲',
    'fear': 'Thriller by Michael Jackson 👻',
    'disgust': 'Bad Guy by Billie Eilish 🤢'
}

# Turn on the webcam (0 is default)
cap = cv2.VideoCapture(0)

# Load Face Detection Model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🎥 Camera starting... Please wait (First time load takes time!)")
print("💡 Tip: Press 'q' to stop the app.")

# --- 2. MAIN LOOP (REAL-TIME PROCESSING) ---
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to Grayscale (Better for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        try:
            # --- AI MAGIC START ---
            
            # Step A: Pre-processing (Crop the face)
            face_img = frame[y:y+h, x:x+w]
            
            # Step B: Emotion Analysis (DeepFace)
            # We enforce_detection=False to prevent crashing if face is blurry
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            
            # Step C: Extract Dominant Emotion
            emotion = result[0]['dominant_emotion']
            
            # Step D: Music Suggestion Logic
            song = music_dict.get(emotion, "Unknown Song")
            print(f"User is {emotion}: Suggesting -> {song}")

            # --- VISUALIZATION ---
            
            # 1. Write Emotion Text
            cv2.putText(frame, emotion.upper(), (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 2. Write Song Recommendation (Below face)
            cv2.putText(frame, f"Song: {song}", (x, y+h+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        except Exception as e:
            pass # Ignore errors if face is not clear

    # Display the resulting frame
    cv2.imshow('Abhishek AI Face Scanner', frame)

    # Quit Logic (Press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped.")