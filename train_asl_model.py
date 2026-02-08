import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
DATASET_PATH = "asl_dataset/"

def extract_hand_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in handLms.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        wrist = handLms.landmark[0]
        tip_ids = [4, 8, 12, 16, 20]
        for i in tip_ids:
            tip = handLms.landmark[i]
            dist = np.linalg.norm([tip.x - wrist.x, tip.y - wrist.y, tip.z - wrist.z])
            landmarks.append(dist)
        return landmarks
    else:
        return None

def load_dataset():
    X = []
    y = []
    for label in labels:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            features = extract_hand_landmarks(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
    return X, y

print("Loading dataset and extracting features...")
X, y = load_dataset()
print(f"Extracted features from {len(X)} images.")

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))

with open("asl_classifier_optimized.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Model training complete and saved.")