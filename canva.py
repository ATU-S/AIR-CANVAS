__author__ = 'ATU-S'

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Hand module
mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # You can change 0 to the camera index you want to use

# Get the camera's frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Calculate the aspect ratio
aspect_ratio = frame_width / frame_height

# Set the desired width for resizing
desired_width = 800  # You can change this value based on your preference

# Calculate the corresponding height to maintain the aspect ratio
desired_height = int(desired_width / aspect_ratio)

# List to store points along the path of the index finger tip
finger_path = []

# List to store colors for each point in the finger path
finger_colors = []

# Color options
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]  # Green, Red, Blue, Yellow
color_names = ['Green', 'Red', 'Blue', 'Yellow']

# Initial color index
current_color_index = 0

# Initial delay time (in seconds)
delay_time = 5  # Increase the delay time

# Initialize start_time
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            landmarks = [(int(landmark.x * frame_width), int(landmark.y * frame_height)) for landmark in hand_landmarks.landmark]

            # Check if the index finger is up
            if landmarks[8][1] < landmarks[7][1]:
                # Check if the thumb and index finger are close together
                thumb_tip = (landmarks[4][0], landmarks[4][1])
                index_tip = (landmarks[8][0], landmarks[8][1])

                # Calculate the distance between thumb and index finger tips
                distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)

                # If the fingers are very close together, switch colors
                if distance < 100:  # Increase the distance threshold
                    # Check if the color has already changed recently
                    if time.time() - start_time > delay_time:
                        current_color_index = (current_color_index + 1) % len(colors)

                        # Reset the delay time
                        start_time = time.time()

                # Add the current index finger tip position to the finger path
                finger_path.append(index_tip)
                finger_colors.append(colors[current_color_index])

                # Draw connections between finger joints
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)

                # Draw the line connecting points along the path with their respective colors
                for i in range(1, len(finger_path)):
                    cv2.line(frame, finger_path[i - 1], finger_path[i], finger_colors[i], 2)
            else:
                # Clear the drawing when the hand is closed
                finger_path = []
                finger_colors = []

    # Display the current color on the frame
    cv2.putText(frame, f'Color: {color_names[current_color_index]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Resize the frame to maintain the aspect ratio
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Display the resulting frame
    cv2.imshow('AIR CANVAS', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
