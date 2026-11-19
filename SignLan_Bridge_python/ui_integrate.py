import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Label dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'P', 15: 'O', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: '10',
    37: 'Who', 38: 'What', 39: 'When', 40: 'Where', 41: 'Which',
    42: 'Hello', 43: 'Goodbye', 44: 'Thank you', 45: 'Sorry', 46: 'Please', 47: 'I love you',
    48: 'Yes', 49: 'No', 50: 'Eat', 51: 'Drink', 52: 'Stop'
}


# Variables
sentence = []
last_word = None
last_prediction_time = time.time()
cap = None  # Camera capture object
running = False  # Control flag for detection

# Tkinter window setup
root = tk.Tk()
root.title("SignLan Bridge")
root.geometry("900x700")
root.configure(bg="#f0f0f0")

# Create a style for ttk Buttons
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10, relief="raised")

# Title Label
lbl_title = tk.Label(root, text="SignLan Bridge", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#333")
lbl_title.pack(pady=10)

# Frame for video feed
frame_video = tk.Frame(root, bg="black", width=700, height=500)
frame_video.pack(pady=10)
lbl_video = tk.Label(frame_video, bg="black")
lbl_video.pack()

# Textbox for detected words
detected_text = tk.StringVar()
txt_detected = tk.Entry(root, textvariable=detected_text, font=("Arial", 14), width=50, state="readonly", justify="center")
txt_detected.pack(pady=10)

# Button Frame
frame_buttons = tk.Frame(root, bg="#f0f0f0")
frame_buttons.pack(pady=10)

btn_start = ttk.Button(frame_buttons, text="🎥 Start Detection", command=lambda: threading.Thread(target=start_detection, daemon=True).start(), style="TButton")
btn_start.grid(row=0, column=0, padx=10, pady=5)

def stop_detection():
    """Stop the sign language detection."""
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    lbl_video.config(image='')  # Clear the video feed

btn_stop = ttk.Button(frame_buttons, text="🛑 Stop Detection", command=stop_detection, style="TButton")
btn_stop.grid(row=0, column=1, padx=10, pady=5)
def start_detection():
    """Start the sign language detection."""
    global cap, running
    if not running:
        running = True
        cap = cv2.VideoCapture(0)
        detect_signs()





def clear_sentence():
    """Clear the detected sentence."""
    global sentence
    sentence = []
    update_textbox()
    
btn_clear = ttk.Button(frame_buttons, text="📝 Clear Sentence", command=clear_sentence, style="TButton")
btn_clear.grid(row=0, column=2, padx=10, pady=5)





def update_textbox():
    """Update the text box with detected words."""
    sentence_text = " ".join(sentence[-10:])
    detected_text.set(sentence_text)


def detect_signs():
    """Real-time hand detection and sentence formation."""
    global cap, running, last_word, last_prediction_time, sentence

    while running:
        if cap is None:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                # Extract features from landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalized x coordinate
                    data_aux.append(y - min(y_))  # Normalized y coordinate

            # Handle One-Hand and Two-Hand Gestures
            num_features = len(data_aux)
            if num_features in [84, 168]:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

                # Prevent duplicate predictions appearing too fast
                current_time = time.time()
                if predicted_character != last_word or (current_time - last_prediction_time > 2):  # 2 sec cooldown
                    last_word = predicted_character
                    last_prediction_time = current_time
                    sentence.append(predicted_character)
                    root.after(0, update_textbox)  # Ensure UI updates safely in the main thread

                # Get bounding box
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)

                # Display predicted word
                cv2.putText(frame, predicted_character,
                            (x1, max(y1 - 10, 30)),  # Ensure text doesn't go off-screen
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        # Convert frame for display in Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update UI in the main thread
        root.after(0, lambda: lbl_video.config(image=imgtk))
        lbl_video.imgtk = imgtk


# Run the application
root.mainloop()
