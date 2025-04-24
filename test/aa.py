import cv2
import mediapipe as mp
import numpy as np

# 얼굴 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# 비디오 캡처 (웹캠 사용)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 이미지를 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 추출
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:  # 얼굴이 인식되었을 경우
        for face_landmarks in results.multi_face_landmarks:
            # 랜드마크 좌표 추출 및 저장
            landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks.landmark])
            np.save('face_landmarks.npy', landmarks)  # 랜드마크를 파일로 저장

            # 랜드마크를 화면에 그리기
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Face Landmark', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

