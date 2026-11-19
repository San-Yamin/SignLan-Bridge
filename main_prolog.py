import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess


# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Labels
labels_dict = {
    0: "Hello", 1: "Goodbye", 2: "Thank you", 3: "Sorry", 4: "Please",
    5: "Excuse me", 6: "Yes", 7: "No", 8: "Help", 9: "I love you",

    10: "I", 11: "You", 12: "He / She / They", 13: "We / Us",
    14: "My / Mine", 15: "Your / Yours", 16: "Who", 17: "What",
    18: "When", 19: "Where",

    20: "Want", 21: "Need", 22: "Go", 23: "Come", 24: "Eat",
    25: "Drink", 26: "Like", 27: "Love", 28: "Stop", 29: "Help",

    30: "Food", 31: "Water", 32: "House", 33: "Car", 34: "Work",
    35: "School", 36: "Friend", 37: "Family", 38: "Home", 39: "Book",

    40: "Big", 41: "Small", 42: "Happy", 43: "Sad", 44: "Fast",
    45: "Slow", 46: "Hot", 47: "Cold"
}

# Variables
sentence = []
cap = None
running = False
frame_skip = 2  # Process every Nth frame
frame_count = 0
last_detected = None  # Store last detected gesture
last_detection_time = 0  # Store last detection timestamp

# Tkinter Setup
root = tk.Tk()
root.title("Sign Language Translator")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)

# Title Label
lbl_title = tk.Label(root, text="Sign Language Translator", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#333")
lbl_title.pack(pady=10)

lbl_video = tk.Label(root, bg="black")
lbl_video.pack(pady=10)

detected_text = tk.StringVar()
txt_detected = tk.Entry(root, textvariable=detected_text, font=("Arial", 14), width=50, state="readonly", justify="center")
txt_detected.pack(pady=10)

validity_text = tk.StringVar()
txt_validity = tk.Label(root, textvariable=validity_text, font=("Arial", 14), width=50, bg="#f0f0f0", anchor="center")
txt_validity.pack(pady=10)

def start_detection():
    """ Start hand gesture detection. """
    global cap, running
    if not running:
        running = True
        cap = cv2.VideoCapture(0)
        threading.Thread(target=detect_signs, daemon=True).start()

def stop_detection():
    """ Stop detection and release camera. """
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    lbl_video.config(image='')

def update_textbox():
    """ Update the detected sentence text box. """
    detected_text.set(" ".join(sentence[-10:]))
    check_validity()  # Check validity every time the sentence updates

def check_validity():
    """ Check the validity of the current sentence using Prolog via subprocess. """
    # Convert sentence into Prolog format (list of atoms)
    prolog_sentence = f"[{', '.join([word.lower() for word in sentence])}]"

    # Log the query being passed to Prolog
    # print(f"Prolog Query: valid_sentence({prolog_sentence})")

    # Run Prolog and capture output
    prolog_query = f"valid_sentence({prolog_sentence})"
    result = run_prolog_query(prolog_query)

    # Update the validity label based on the result
    if result == True:
        validity_text.set("Valid Sentence")
        txt_detected.config(fg="green")
    else:
        validity_text.set("Not a Valid Sentence")
        txt_detected.config(fg="red")


def run_prolog_query(query):
    try:
        prolog_script = f"consult('D:/UIT/3rd yr 1st sem/Artificial Intelligence(5105)/Project/sign-language_v2/validity_check.pl'), {query}, write('Result: '), writeln(true)."

        result = subprocess.run(
            ["swipl", "-q", "-g", prolog_script, "-t", "halt"],
            capture_output=True, text=True
        )

        print(f"Prolog Raw Output: {result.stdout.strip()}")
        print(f"Prolog Error (if any): {result.stderr.strip()}")

        if result.stderr:
            print(f"Prolog Error: {result.stderr.strip()}")
            return "error"

        # Check for 'Result: true' or 'Result: false'
        if "Result: true" in result.stdout:
            return True
        elif "Result: false" in result.stdout:
            return False
        else:
            return "error"
    
    except Exception as e:
        print(f"Error running Prolog: {e}")
        return "error"
    
def detect_signs():
    """ Hand gesture detection loop. """
    global cap, running, frame_count, last_detected, last_detection_time  # Declare global variables
    
    while running:
        if cap is None:
            break
        
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to reduce lag

        frame = cv2.resize(frame, (500, 350))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_, y_ = [], []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            if len(data_aux) == 42:
                try:
                    prediction = model.predict_proba([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(np.argmax(prediction[0]), "Unknown")
                    confidence = max(prediction[0])

                    print(f"Predicted: {predicted_character}, Confidence: {confidence:.2f}")

                    # ✅ FIX: Use global variables
                    if confidence > 0.7 and (last_detected is None or predicted_character != last_detected or time.time() - last_detection_time > 2):
                        last_detected = predicted_character
                        last_detection_time = time.time()
                        sentence.append(predicted_character)
                        root.after(0, update_textbox)

                except AttributeError:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

                    # ✅ FIX: Use global variables
                    if last_detected is None or predicted_character != last_detected or time.time() - last_detection_time > 2:
                        last_detected = predicted_character
                        last_detection_time = time.time()
                        sentence.append(predicted_character)
                        root.after(0, update_textbox)

        # Convert frame to display in Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        root.after(0, lambda: lbl_video.config(image=imgtk))
        lbl_video.imgtk = imgtk

        time.sleep(0.03)  # Small delay to reduce lag


# Buttons
frame_buttons = tk.Frame(root, bg="#f0f0f0")
frame_buttons.pack(pady=10)

ttk.Button(frame_buttons, text="🎥 Start Detection", command=start_detection).grid(row=0, column=0, padx=10)
ttk.Button(frame_buttons, text="🛑 Stop Detection", command=stop_detection).grid(row=0, column=1, padx=10)

def clear_sentence():
    """ Clear detected sentence. """
    global sentence
    sentence = []
    update_textbox()

ttk.Button(frame_buttons, text="📝 Clear Sentence", command=clear_sentence).grid(row=0, column=2, padx=10)

def undo_last_word():
    """ Remove last word from sentence. """
    global sentence
    if sentence:
        sentence.pop()
        update_textbox()

ttk.Button(frame_buttons, text="↩️ Undo Last", command=undo_last_word).grid(row=0, column=3, padx=10)

root.mainloop()
