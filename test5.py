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

# 손 랜드마크 그리기용
mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처
cap = cv2.VideoCapture(0)
prev_points = [None, None]  # 양손의 이전 위치 저장
canvas = None

# 손 라벨에 따라 색 설정
colors = {'Left': (0, 255, 0), 'Right': (255, 0, 0)}

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
        # 이전 포인트 배열 크기 조정
        if len(prev_points) < len(result.multi_hand_landmarks):
            prev_points.extend([None] * (len(result.multi_hand_landmarks) - len(prev_points)))

        for idx, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            hand_label = handedness.classification[0].label  # 'Left' 또는 'Right'
            color = colors[hand_label]

            # 손가락 끝 좌표 (검지)
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # 지우개 기능: 엄지(4번)와 검지(8번) 사이 거리 측정
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            distance = np.linalg.norm(
                np.array([thumb.x - index.x, thumb.y - index.y])
            )

            if distance < 0.05:  # 손가락 두 개가 가까우면
                cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)  # 검정 원으로 지우기
                prev_points[idx] = None
            else:
                if prev_points[idx] is not None:
                    cv2.line(canvas, prev_points[idx], (x, y), color, 4)

                prev_points[idx] = (x, y)

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img = cv2.add(img, canvas)
    cv2.imshow('Drawing with Hands', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
