import os
import cv2
import numpy as np

IMAGE_SIZE = 64
SEQUENCE_LENGTH = 20
MAX_VIDEOS_PER_CLASS = 60  # keep CPU friendly

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


def create_dataset(data_dir):
    features = []
    labels = []

    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        print(f"\nProcessing class: {class_name}")

        video_files = os.listdir(class_path)[:MAX_VIDEOS_PER_CLASS]

        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)

            frames = extract_frames(video_path)

            if frames is not None:
                features.append(frames)
                labels.append(label)

    return np.array(features), np.array(labels), class_names