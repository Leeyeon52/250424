import cv2
import mediapipe as mp
import numpy as np
import os  # os 모듈을 import 해야 합니다
import time
import pickle

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # 실시간 모드
    max_num_faces=1,
    refine_landmarks=True
)

# 얼굴 벡터 리스트
reference_faces = []

# 저장된 얼굴 벡터 불러오기 (이미 등록된 벡터가 있으면 이를 불러옴)
if os.path.exists('face_vectors.pkl'):
    with open('face_vectors.pkl', 'rb') as f:
        reference_faces = pickle.load(f)

# 실시간 얼굴 인식 및 구분
cap = cv2.VideoCapture(0)
prev_time = time.time()
diff_count = 0
MAX_DIFF_COUNT = 10

# 얼굴 등록 모드 여부
registering = False

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
        current_landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])
        current_face_vector = current_landmarks.flatten()

        matched = False
        for ref_vector in reference_faces:
            diff = np.linalg.norm(ref_vector - current_face_vector)
            if diff < 0.04:  # 임계값을 조정하여 매칭 강도를 결정
                matched = True
                break

        if not matched:
            diff_count += 1
            print(f"❌ 허용되지 않은 얼굴! ({diff_count}/{MAX_DIFF_COUNT})")
            if diff_count >= MAX_DIFF_COUNT:
                print("🚫 얼굴 인증 실패. 프로그램 종료.")
                break
        else:
            diff_count = 0
            cv2.putText(frame, "✔ 인증 성공", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 얼굴 등록 모드 활성화
        if registering:
            # 등록된 얼굴 벡터 저장
            reference_faces.append(current_face_vector)
            with open('face_vectors.pkl', 'wb') as f:
                pickle.dump(reference_faces, f)
            print("📸 얼굴 등록 완료!")
            registering = False

        # 키 입력으로 등록 모드 활성화
        cv2.putText(frame, "Press 'r' to register face", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # FPS 출력
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 'r' 키를 눌러 얼굴 등록 모드 활성화
    if cv2.waitKey(1) & 0xFF == ord('r'):
        registering = True

    cv2.imshow('Face Image Auth Check', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
