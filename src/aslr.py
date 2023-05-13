import pickle

import cv2
import mediapipe as mp
import numpy as np

MODEL_DIR = '../data/model.p'

with open(MODEL_DIR, 'rb') as f:
    model = pickle.load(f)
    model = model['model']

webcam = cv2.VideoCapture(cv2.CAP_ANY)


hands_modeling = mp.solutions.hands
hands_rendering = mp.solutions.drawing_utils
rendering_styles = mp.solutions.drawing_styles

hands = hands_modeling.Hands(static_image_mode=True, min_detection_confidence=0.3)

label_mapping = {i: model.classes_[i] for i in range(len(model.classes_))}

while(True):
    entry = []
    landmark_xs = []
    landmark_ys = []

    _, frame_bgr = webcam.read()

    height, width, _ = frame_bgr.shape

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)

    if not result.multi_hand_landmarks:
        cv2.imshow('frame', frame_bgr)
        cv2.waitKey(1)
        continue

    # Assume only one hand (if two or more shown, only getting the first one)
    landmarks = result.multi_hand_landmarks[0]

    # Drawing skeletal landmarks over frame
    hands_rendering.draw_landmarks(
        frame_bgr,
        landmarks,
        hands_modeling.HAND_CONNECTIONS,
        rendering_styles.get_default_hand_landmarks_style(),
        rendering_styles.get_default_hand_connections_style()
        )
    
    for lm in landmarks.landmark:
        landmark_xs.append(lm.x)
        landmark_ys.append(lm.y)
    
    for lm in landmarks.landmark:
        entry.append(lm.x - min(landmark_xs))
        entry.append(lm.y - min(landmark_ys))

    # Bounding box around hand
    x_min_bound = int(min(landmark_xs) * width) - 20
    x_max_bound = int(max(landmark_xs) * width) + 20

    y_min_bound = int(min(landmark_ys) * height) - 20
    y_max_bound = int(max(landmark_ys) * height) + 20

    cv2.rectangle(frame_bgr, (x_min_bound, y_min_bound),
                  (x_max_bound, y_max_bound), (255,255,255), 4)

    # Predicted letter
    prediction = model.predict([np.asarray(entry)])[0]

    cv2.putText(frame_bgr, prediction, (x_min_bound, y_min_bound - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame_bgr)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()