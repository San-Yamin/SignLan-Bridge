import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Load pre-trained model using pickle
# You need to replace 'model.pkl' with the actual filename
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand tracker
hand_tracker = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize variables
gesture_stable_counter = 0
gesture_threshold = 5  # Number of frames to stabilize the gesture
previous_gesture = None
current_gesture = None

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Tkinter Setup
root = tk.Tk()
root.title("Sign Language Recognition")

# Canvas to display video frames in Tkinter
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

def process_frame():
    global gesture_stable_counter, previous_gesture, current_gesture

    ret, frame = cap.read()
    if not ret:
        return

    # Resize frame for lower resolution and reduce sensitivity
    frame = cv2.resize(frame, (640, 480))

    # Apply Gaussian Blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    results = hand_tracker.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture detection logic (simplified; replace with actual model inference)
            # Example: If hand landmarks indicate a specific gesture, set current_gesture
            current_gesture = "Example Gesture"  # Replace with actual gesture detection logic

            if current_gesture == previous_gesture:
                gesture_stable_counter += 1
            else:
                gesture_stable_counter = 0

            if gesture_stable_counter >= gesture_threshold:
                # Trigger action (e.g., print gesture or perform action)
                print(f"Gesture detected: {current_gesture}")

            # Update previous gesture
            previous_gesture = current_gesture

    # Convert frame back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Convert the frame to Image for Tkinter display
    img = Image.fromarray(frame_bgr)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the canvas with the new frame
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk

    # Call the function again after 30 ms
    root.after(30, process_frame)

def on_close():
    # Release video capture and close the Tkinter window
    cap.release()
    root.quit()

# Run the video capture and GUI loop
root.protocol("WM_DELETE_WINDOW", on_close)
process_frame()

# Start Tkinter main loop
root.mainloop()
