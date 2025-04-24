import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys

# 기준 손 랜드마크 로드
try:
    reference_landmarks = np.load('hand_landmarks.npy')
except FileNotFoundError:
    print("⚠️ 기준 손 데이터(hand_landmarks.npy)가 없습니다. 먼저 저장해 주세요.")
    sys.exit()

# Mediapipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

prev_time = time.time()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    overlay = frame.copy()
    dark_bg = (frame * 0.2).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # 현재 손 좌표 추출 (x, y만)
        current_landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

        # 기준과 비교
        if reference_landmarks.shape == current_landmarks.shape:
            diff = np.linalg.norm(reference_landmarks - current_landmarks)
            print(f"손 차이값: {diff:.4f}")

            if diff > 0.06:
                print("❌ 손이 기준과 다릅니다. 프로그램을 종료합니다.")
                break
        else:
            print("⚠️ 손 랜드마크 개수가 일치하지 않습니다.")
            break

        # 시각화용 포인트
        points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark], dtype=np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

        for (x, y) in points:
            cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)

        # 손 구조 그리기
        mp_drawing.draw_landmarks(
            overlay,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # 배경 어둡게, 손만 밝게
    mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_blur = cv2.merge([mask_blur] * 3) / 255.0
    result = (overlay.astype(np.float32) * mask_blur + dark_bg.astype(np.float32) * (1 - mask_blur)).astype(np.uint8)

    # FPS 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(result, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Hand Verification', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
