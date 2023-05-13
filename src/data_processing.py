import os
import pickle

import mediapipe as mp
import cv2

RAW_DATA_DIR = '../data/raw_data'


hands_modeling = mp.solutions.hands
hands_rendering = mp.solutions.drawing_utils
rendering_styles = mp.solutions.drawing_styles

hands = hands_modeling.Hands(static_image_mode=True, min_detection_confidence=0.3)


data = []
labels = []

# Converts the input images into coordinate entries for each of the Hands anchor points
for label_dir in os.listdir(RAW_DATA_DIR):
    for img_path in os.listdir(os.path.join(RAW_DATA_DIR, label_dir)):
        # openCV's imread uses BGR ordering for their images
        # MediaPipe Hands uses RGB ordering
        img_bgr = cv2.imread(os.path.join(RAW_DATA_DIR, label_dir, img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        entry = []

        landmark_xs = []
        landmark_ys = []

        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            # Model was not confident in identifying a hand (or no hand was present)
            continue
        if len(result.multi_hand_landmarks) !=  1:
            # Model found two hands
            continue
        
        landmarks = result.multi_hand_landmarks[0]

        # Cycling through the landmarks to store coordinates
        for lm in landmarks.landmark:
            landmark_xs.append(lm.x)
            landmark_ys.append(lm.y)

        # Saving relative positions of landmarks as data entry
        for lm in landmarks.landmark:
            entry.append(lm.x - min(landmark_xs))
            entry.append(lm.y - min(landmark_ys))
            
        data.append(entry)
        labels.append(label_dir)

with open('../data/data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

