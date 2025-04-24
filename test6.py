import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처
cap = cv2.VideoCapture(0)
prev_points = [None, None]
canvas = None

# 손 모드: 기본 False
eraser_mode = [False, False]

# 손 라벨에 따라 색 설정 (왼손: 초록, 오른손: 진한 파랑)
colors = {'Left': (0, 255, 0), 'Right': (255, 50, 0)}  # 진한 파란색 계열

def is_hand_open(hand_landmarks):
    # 손바닥 펴졌는지 판별: 손가락 끝이 손바닥보다 위에 있으면 열림
    fingers = [8, 12, 16, 20]
    open_count = 0
    for idx in fingers:
        if hand_landmarks.landmark[idx].y < hand_landmarks.landmark[idx - 2].y:
            open_count += 1
    return open_count >= 3

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    if canvas is None:
        canvas = np.zeros_like(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        if len(prev_points) < len(result.multi_hand_landmarks):
            prev_points.extend([None] * (len(result.multi_hand_landmarks) - len(prev_points)))
            eraser_mode.extend([False] * (len(result.multi_hand_landmarks) - len(eraser_mode)))

        for idx, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            label = handedness.classification[0].label
            color = colors[label]

            # 손바닥 열렸는지 판단 → 토글
            if is_hand_open(hand_landmarks):
                eraser_mode[idx] = not eraser_mode[idx]
                prev_points[idx] = None  # 손 떼고 다시 시작
                continue  # 토글 시 바로 패스 (연속 인식 방지)

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            if eraser_mode[idx]:
                cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                prev_points[idx] = None
            else:
                if prev_points[idx] is not None:
                    cv2.line(canvas, prev_points[idx], (x, y), color, 4)
                prev_points[idx] = (x, y)

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img = cv2.add(img, canvas)
    cv2.imshow('Gesture Drawing with Toggle', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
