import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    overlay = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 외곽 좌표 수집
            points = np.array(
                [[int(landmark.x * w), int(landmark.y * h)] for landmark in face_landmarks.landmark],
                dtype=np.int32
            )

            # 얼굴 전체를 포함하는 마스크 생성
            cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

            # 얼굴에 초록 점 표시
            for (x, y) in points:
                cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

    # 배경 어둡게 처리
    dark_bg = (frame * 0.2).astype(np.uint8)
    result = np.where(mask[..., None] == 255, overlay, dark_bg)

    cv2.imshow('Face Focus', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
