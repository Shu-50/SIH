
import cv2
import mediapipe as mp
import joblib
import numpy as np
from googletrans import Translator
from gtts import gTTS
import os
import pyttsx3

# Initialize translation and text-to-speech
translator = Translator()
tts_engine = pyttsx3.init()

# Load the trained model, scaler, and label encoder
clf = joblib.load('models/gesture_classifier.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Initialize Mediapipe Hands and drawing utility
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam feed
cap = cv2.VideoCapture(0)

# Initialize lists to store confirmed gestures and sentences
confirmed_gestures = []
current_gesture = None
sentence = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect hand landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert landmarks to numpy array for model prediction
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks_normalized = scaler.transform(landmarks)  # Normalize landmarks

            # Predict the gesture
            prediction = clf.predict(landmarks_normalized)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Display the prediction on the frame
            cv2.putText(frame, f'Gesture: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Update current gesture
            current_gesture = predicted_label

    # Display the current sentence
    if sentence:
        sentence_text = ' '.join(sentence)
        cv2.putText(frame, f'Sentence: {sentence_text}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Sentence: ', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Gesture Detection', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit the loop
        break
    elif key == ord('c'):  # Confirm the current gesture
        if current_gesture:
            confirmed_gestures.append(current_gesture)
            sentence.append(current_gesture)
            current_gesture = None  # Reset current gesture
    elif key == ord('r'):  # Reset the current word
        confirmed_gestures = []
        current_gesture = None
        sentence = []  # Reset the entire sentence
    elif key == ord(' '):  # Add space between words
        sentence.append(' ')  # Insert a space in the sentence

# Final sentence creation
final_sentence = ''.join(sentence).replace('  ', ' ').strip()  # Strip extra spaces
print(f"Final Sentence: {final_sentence}")

# Translate to Hindi (Meaning-based translation)
translated_sentence = translator.translate(final_sentence, src='en', dest='hi').text
print(f"Translated Sentence (Hindi): {translated_sentence}")

# Convert to speech
def text_to_speech(text, lang='en'):
    audio_file = 'output.mp3'
    
    if lang == 'hi':
        tts = gTTS(text=text, lang='hi')
    else:
        tts = gTTS(text=text, lang='en')

    tts.save(audio_file)
    os.system(f"start {audio_file}")  # For Windows; use "open" on Mac or "xdg-open" on Linux.

# Speak the sentences
print("Playing English Audio...")
text_to_speech(final_sentence, lang='en')

print("Playing Hindi Audio...")
text_to_speech(translated_sentence, lang='hi')

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
