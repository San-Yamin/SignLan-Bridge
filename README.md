# SignLan Bridge

**SignLan Bridge** is a computer-vision sign language translator developed by Group 8 for CST-5105: Artificial Intelligence at the University of Information Technology (academic year 2024–2025).

The application uses a webcam to detect hand landmarks with [MediaPipe](https://mediapipe.dev/), classify gestures with a trained Random Forest model, and display the detected words in a desktop Tkinter interface. An optional SWI-Prolog integration checks recognized words against a small set of valid sentence patterns.

## Features

- Real-time webcam hand detection
- MediaPipe hand-landmark feature extraction
- Random Forest gesture classification
- Detected-word sentence composition
- Start, stop, clear, and undo controls
- Optional Prolog-based sentence validity checking
- Tools for collecting images, creating feature data, and retraining the model

## Project structure

```text
.
├── SignLan_Bridge_python/
│   ├── main.py                 # Main translator UI
│   ├── main_prolog.py          # Translator UI with Prolog validation
│   ├── collect_imgs.py         # Capture training images
│   ├── create_dataset.py       # Extract hand-landmark features
│   ├── train_classifier.py     # Train and save the Random Forest model
│   ├── inference_classifier.py # Webcam inference example
│   └── final.py                # Earlier UI prototype
├── data.pickle                 # Extracted training features
├── model.p                    # Trained classifier
├── main_prolog.py              # Root-level Prolog-enabled prototype
└── validity_check.pl           # Valid sentence patterns
```

The raw `data/` image dataset is intentionally excluded from Git because it is approximately 390 MB. Keep it locally when retraining the model.

## Requirements

- Python 3.10 or 3.11
- A working webcam
- macOS, Windows, or Linux with Tkinter installed
- Optional: SWI-Prolog for sentence validation

Install the Python packages:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the translator

From the project root:

```bash
python SignLan_Bridge_python/main.py
```

The program loads `model.p`, opens the webcam after **Start Detection** is selected, and appends sufficiently confident recognized signs to the sentence display.

## Retrain the model

To create a new dataset, extract features, and train a new model:

```bash
python SignLan_Bridge_python/collect_imgs.py
python SignLan_Bridge_python/create_dataset.py
python SignLan_Bridge_python/train_classifier.py
```

These commands expect a local `data/` directory containing one numbered subdirectory per gesture class. Retraining overwrites `data.pickle` and `model.p`.

## Prolog validation

`validity_check.pl` contains the sentence patterns used by the Prolog-enabled prototype. SWI-Prolog must be installed and available as `swipl` on your `PATH`.

The Prolog integration currently contains a machine-specific path in `main_prolog.py`; update that path to the location of your local `validity_check.pl` before running it.

## Limitations

- The trained labels and supported vocabulary are defined directly in the Python scripts.
- Recognition accuracy depends on lighting, camera quality, hand position, and the training data.
- This is an academic prototype and is not a complete natural-language translation system.

## Academic information

- **Project:** SignLan Bridge: Sign Language Translator
- **Course:** CST-5105: Artificial Intelligence
- **Institution:** University of Information Technology
- **Team:** Group 8
- **Date:** 10 March 2025
