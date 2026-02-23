import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

IMAGE_SIZE = 64
SEQUENCE_LENGTH = 20

model = load_model("har_model.h5")

# Replace this list with your actual class names
class_names = [
    "Basketball",
    "Biking",
    "JumpingJack",
    "PushUps",
    "WalkingWithDog"
]

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return None

    skip = max(total_frames // SEQUENCE_LENGTH, 1)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == SEQUENCE_LENGTH:
        return np.array(frames)

    return None

def predict(file):
    # Gradio returns dictionary in HF
    if file is None:
        return "No file uploaded"
    
    video_path = file.name

    frames = extract_frames(video_path)

    if frames is None:
        return "Error processing video."

    frames = frames.reshape(1, SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)

    prediction = model.predict(frames)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return f"Prediction: {class_names[predicted_class]} | Confidence: {confidence:.2f}%"

demo = gr.Interface(
    fn=predict,
    inputs=gr.File(file_types=["video"]),
    outputs="text",
    title="Human Action Recognition (CNN + LSTM)",
    description="Upload a short video clip (≤10 seconds)"
)

demo.launch()