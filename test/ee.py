import cv2
import mediapipe as mp
import numpy as np
import os
import time
import pickle

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 저장된 얼굴 벡터 불러오기
reference_faces = []
if os.path.exists('face_vectors.pkl'):
    with open('face_vectors.pkl', 'rb') as f:
        reference_faces = pickle.load(f)

# 웹캠 열기
cap = cv2.VideoCapture(0)
prev_time = time.time()
diff_count = 0
MAX_DIFF_COUNT = 10
registering = False
register_success_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❗ 웹캠에서 프레임을 읽지 못했습니다. 다시 시도 중...")
        continue

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        current_landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])
        current_face_vector = current_landmarks.flatten()

        matched = False
        for ref_vector in reference_faces:
            diff = np.linalg.norm(ref_vector - current_face_vector)
            if diff < 0.04:
                matched = True
                break

        if matched:
            diff_count = 0
            cv2.putText(frame, "✔ 인증 성공", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            diff_count += 1
            cv2.putText(frame, f"❌ 인증 실패 ({diff_count}/{MAX_DIFF_COUNT})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            if diff_count >= MAX_DIFF_COUNT:
                cv2.putText(frame, "🚫 인증 실패 - 등록 필요", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 등록 처리
        if registering:
            reference_faces.append(current_face_vector)
            with open('face_vectors.pkl', 'wb') as f:
                pickle.dump(reference_faces, f)
            register_success_time = time.time()
            registering = False
            print("📸 얼굴 등록 완료!")

    # 등록 성공 메시지 표시
    if register_success_time and (time.time() - register_success_time < 2):
        cv2.putText(frame, "📸 등록 완료!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 128, 255), 3)
    else:
        if not registering:
            cv2.putText(frame, "Press 'r' to register face", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # FPS 출력
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        registering = True
    elif key == ord('q'):
        break

    cv2.imshow('Face Registration & Auth', frame)

cap.release()
cv2.destroyAllWindows()
