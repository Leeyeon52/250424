import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys

# 기준 얼굴 좌표 불러오기
try:
    reference_landmarks = np.load('face_landmarks.npy')
except FileNotFoundError:
    print("⚠️ 기준 얼굴 데이터(face_landmarks.npy)가 없습니다. 먼저 저장해 주세요.")
    sys.exit()

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

prev_time = time.time()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    overlay = frame.copy()
    dark_bg = (frame * 0.2).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # 현재 얼굴 랜드마크 추출
        current_landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])

        # 차이 계산 (Euclidean distance)
        if reference_landmarks.shape == current_landmarks.shape:
            diff = np.linalg.norm(reference_landmarks - current_landmarks)
            print(f"차이값: {diff:.4f}")

            # 기준 이상 다르면 종료
            if diff > 0.04:  # 민감도 조절 가능 (0.03~0.06 권장)
                print("❌ 얼굴이 기준과 다릅니다. 프로그램을 종료합니다.")
                break
        else:
            print("⚠️ 얼굴 랜드마크 개수가 일치하지 않습니다.")
            break

        # 얼굴 메쉬 및 시각화
        points = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark],
            dtype=np.int32
        )
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

        for (x, y) in points:
            cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

        mp_drawing.draw_landmarks(
            image=overlay,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
        )

    # 배경 블러
    mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_blur = cv2.merge([mask_blur] * 3) / 255.0
    result = (overlay.astype(np.float32) * mask_blur + dark_bg.astype(np.float32) * (1 - mask_blur)).astype(np.uint8)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(result, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Face Auth Check', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
