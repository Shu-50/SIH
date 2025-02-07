

import cv2
import mediapipe as mp
import csv
import os

# Initialize Mediapipe Hands and drawing utility
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# File path to store dataset
dataset_path = 'text to audio/model3/dataset/isl_data.csv'

# Create dataset directory if it doesn't exist
os.makedirs('model3/dataset', exist_ok=True)

# Check if the dataset file exists and add headers if it's empty
file_exists = os.path.isfile(dataset_path)

# List of headers for 21 landmarks (x, y, z for each) and the label
headers = []
for i in range(1, 22):  # 21 landmarks
    headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
headers.append('label')  # Last column for the label

# Write headers if the file doesn't exist or is empty
if not file_exists or os.stat(dataset_path).st_size == 0:
    with open(dataset_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# Open the webcam feed
cap = cv2.VideoCapture(0)

while True:
    print("\n--- ISL Data Collection ---")
    label = input("Enter the label for the gesture (e.g., hello, thank_you) or type 'exit' to quit: ")

    if label.lower() == 'exit':
        break

    # Ask how many samples for this label
    num_samples = int(input(f"How many samples of '{label}' do you want to collect? "))

    sample_count = 0

    # Loop to collect data for the specified number of samples
    while sample_count < num_samples:
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

                # Append the label at the end
                landmarks.append(label)

                # Save the sample to the CSV file when 's' is pressed
                cv2.putText(frame, 'Press s to save gesture, q to quit.', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Collecting Data', frame)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    # Open file in append mode to add the new gesture
                    with open(dataset_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(landmarks)
                        sample_count += 1
                        print(f"Sample {sample_count}/{num_samples} for '{label}' saved.")

        # Exit the collection loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Finished collecting {sample_count} samples for '{label}'.")

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import csv
# import os

# # Initialize Mediapipe Hands and drawing utility
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # File path to store dataset
# dataset_path = 'text to audio/model3/dataset/isl_data.csv'

# # Create dataset directory if it doesn't exist
# os.makedirs('model3/dataset', exist_ok=True)

# # Check if the dataset file exists and add headers if it's empty
# file_exists = os.path.isfile(dataset_path)

# # List of headers for 21 landmarks (x, y, z for each) for both hands and the label
# headers = []
# for hand in ['left', 'right']:
#     for i in range(1, 22):  # 21 landmarks for each hand
#         headers.extend([f'{hand}_landmark_{i}_x', f'{hand}_landmark_{i}_y', f'{hand}_landmark_{i}_z'])
# headers.append('label')  # Last column for the label

# # Write headers if the file doesn't exist or is empty
# if not file_exists or os.stat(dataset_path).st_size == 0:
#     with open(dataset_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)

# # Open the webcam feed
# cap = cv2.VideoCapture(0)

# while True:
#     print("\n--- ISL Data Collection ---")
#     label = input("Enter the label for the gesture (e.g., hello, thank_you) or type 'exit' to quit: ")

#     if label.lower() == 'exit':
#         break

#     # Ask how many samples for this label
#     num_samples = int(input(f"How many samples of '{label}' do you want to collect? "))

#     sample_count = 0

#     # Loop to collect data for the specified number of samples
#     while sample_count < num_samples:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Flip frame for mirror view
#         frame = cv2.flip(frame, 1)

#         # Convert the BGR image to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Process the frame and detect hands
#         results = hands.process(rgb_frame)

#         # Draw hand landmarks if hands are detected
#         if results.multi_hand_landmarks:
#             landmarks = []

#             # Initialize empty lists for left and right hand landmarks
#             left_hand_landmarks = ['NaN'] * 63  # 21 landmarks * 3 coordinates (x, y, z)
#             right_hand_landmarks = ['NaN'] * 63

#             for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                 # Determine which hand it is (0 for right, 1 for left) and collect the landmarks
#                 hand_type = results.multi_handedness[idx].classification[0].label.lower()

#                 hand_data = []
#                 for lm in hand_landmarks.landmark:
#                     hand_data.extend([lm.x, lm.y, lm.z])

#                 # Assign landmarks to the correct hand (left or right)
#                 if hand_type == 'left':
#                     left_hand_landmarks = hand_data
#                 else:
#                     right_hand_landmarks = hand_data

#                 # Draw hand landmarks on the frame
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Combine both hands' landmarks
#             landmarks = left_hand_landmarks + right_hand_landmarks

#             # Append the label at the end
#             landmarks.append(label)

#             # Display the frame with instructions
#             cv2.putText(frame, 'Press s to save gesture, q to quit.', (10, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.imshow('Collecting Data', frame)

#             # Save the sample to the CSV file when 's' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('s'):
#                 with open(dataset_path, mode='a', newline='') as f:
#                     writer = csv.writer(f)
#                     writer.writerow(landmarks)
#                     sample_count += 1
#                     print(f"Sample {sample_count}/{num_samples} for '{label}' saved.")

#         # Exit the collection loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     print(f"Finished collecting {sample_count} samples for '{label}'.")

# # Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()
