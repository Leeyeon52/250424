# 손의 랜드마크를 기준으로 저장하는 코드
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

cap = cv2.VideoCapture(0)
print("손을 카메라에 잘 보이게 하고 아무 키나 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Press any key to capture hand', frame)
    if cv2.waitKey(1) != -1:
        break

h, w, _ = frame.shape
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(rgb)

if results.multi_hand_landmarks:
    landmarks = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    np.save('hand_landmarks.npy', coords)
    print("✅ 손 기준 좌표 저장 완료: hand_landmarks.npy")
else:
    print("❌ 손이 감지되지 않았습니다.")

cap.release()
cv2.destroyAllWindows()
