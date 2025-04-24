import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 저장 폴더 생성
os.makedirs('snapshots', exist_ok=True)

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# FPS 계산용
prev_time = time.time()

# 웹캠 열기
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
    mask_total = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 점 좌표
            points = np.array(
                [[int(landmark.x * w), int(landmark.y * h)] for landmark in face_landmarks.landmark],
                dtype=np.int32
            )

            # 얼굴 마스크 누적
            cv2.fillConvexPoly(mask_total, cv2.convexHull(points), 255)

            # 랜드마크 점
            for (x, y) in points:
                cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

            # 메쉬 연결
            mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )

    # 마스크 경계 부드럽게 (블러 처리)
    mask_blur = cv2.GaussianBlur(mask_total, (51, 51), 0)
    mask_blur = cv2.merge([mask_blur] * 3) / 255.0

    # 최종 출력 이미지 (마스크로 자연스럽게 혼합)
    result = (overlay.astype(np.float32) * mask_blur + dark_bg.astype(np.float32) * (1 - mask_blur)).astype(np.uint8)

    # FPS 계산 및 출력
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(result, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Enhanced Face Mesh', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = int(time.time())
        filename = f'snapshots/face_snapshot_{timestamp}.png'
        cv2.imwrite(filename, result)
        print(f'Snapshot saved: {filename}')

cap.release()
cv2.destroyAllWindows()
