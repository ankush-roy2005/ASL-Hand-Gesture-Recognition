import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Load classifier for ASL
with open('asl_classifier_optimized.pkl', 'rb') as f:
    clf = pickle.load(f)

# Setup MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Audio setup for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Volume smoothing
vol_smooth = 0
vol_threshold = 0.02

# Modes
mode = 0  # 0 = Volume Control, 1 = ASL Recognition
mode_names = ["Volume Control", "ASL Sign Recognition"]

# Variables for mode switching
last_hand_state = None
last_switch_time = 0
mode_switch_cooldown = 0.8

# Sentence building
sentence = ""
current_prediction = ""
last_prediction_time = 0
prediction_stable_threshold = 0.5 # seconds
confirmed_char = ""

# -------------------- FUNCTIONS --------------------
def count_fingers(lmList, handType):
    fingers = []
    # Thumb
    if handType == "Right":
        fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)
    else:
        fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
    # Other fingers
    tipIds = [8, 12, 16, 20]
    for i in tipIds:
        fingers.append(1 if lmList[i][2] < lmList[i - 2][2] else 0)
    return fingers


def extract_landmarks_from_frame(handLms):
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


def is_hand_open(fingers):
    return sum(fingers[1:]) >= 4


def is_hand_closed(fingers):
    return sum(fingers[1:]) == 0


def delete_last_letter(sentence):
    if len(sentence) > 0:
        return sentence[:-1]
    return sentence

# -------------------- MAIN LOOP --------------------
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []
    hand_label = None
    
    # Detect hands and landmarks
    if results.multi_hand_landmarks:
        for i, (handLms, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            hand_label = handedness.classification[0].label
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

    # Mode switching gesture
    if len(lmList) >= 21 and hand_label is not None:
        fingers = count_fingers(lmList, hand_label)
        now = time.time()
        if is_hand_closed(fingers):
            if last_hand_state != "closed":
                last_hand_state = "closed"
                last_switch_time = now
        elif is_hand_open(fingers):
            if last_hand_state == "closed" and (now - last_switch_time > mode_switch_cooldown):
                mode = (mode + 1) % 2
                last_hand_state = "open"
                last_switch_time = now
            else:
                last_hand_state = "open"
        else:
            last_hand_state = None

    # ---- MODE 0: VOLUME CONTROL ----
    if mode == 0 and len(lmList) >= 21:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        targetVolScalar = np.interp(length, [20, 180], [0.0, 1.0])
        currentVol = volume.GetMasterVolumeLevelScalar()
        if abs(currentVol - targetVolScalar) > vol_threshold:
            volume.SetMasterVolumeLevelScalar(targetVolScalar, None)
            vol_smooth = targetVolScalar
        else:
            vol_smooth = currentVol

        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
        vol_bar_pos = int(np.interp(vol_smooth, [0, 1], [400, 150]))
        cv2.rectangle(img, (50, vol_bar_pos), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(vol_smooth * 100)} %', (40, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    # ---- MODE 1: ASL RECOGNITION ----
    if mode == 1 and results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        landmarks = extract_landmarks_from_frame(handLms)
        landmarks_np = np.array(landmarks).reshape(1, -1)
        prediction = clf.predict(landmarks_np)[0]
        
        # Check for stable prediction
        now = time.time()
        if prediction == current_prediction:
            if (now - last_prediction_time) > prediction_stable_threshold:
                # Prediction has been stable for long enough, prepare to confirm
                confirmed_char = prediction
        else:
            current_prediction = prediction
            last_prediction_time = now
            confirmed_char = ""
        
        # Display the potential prediction on the screen
        cv2.putText(img, f"Predicted: {current_prediction}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show sentence at bottom
    cv2.putText(img, f"Sentence: {sentence}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show mode
    cv2.putText(img, f"Mode: {mode_names[mode]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Control + ASL", img)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence = ""
    if key == ord('d'):
        sentence = delete_last_letter(sentence)
    
    # Confirm prediction with 'k' key
    if mode == 1 and key == ord('k'):
        if confirmed_char:
            sentence += confirmed_char
            confirmed_char = ""
            current_prediction = ""
            last_prediction_time = time.time() # Reset timer to prevent immediate re-addition

cap.release()
cv2.destroyAllWindows()