import cv2
import numpy as np
import mediapipe as mp
import random
import os

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 동물 이미지 로딩
animal_images = {}
animal_names = ['dog', 'cat', 'panda', 'fox', 'dino']
for name in animal_names:
    path = f'images/{name}.png'
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함
    if image is None:
        print(f"Error loading {path}!")
    else:
        animal_images[name] = image
        print(f"{name} image loaded successfully.")

# 랜덤 동물 초기 설정
if animal_images:
    selected_animal = random.choice(list(animal_images.keys()))
else:
    print("No animal images loaded.")
    exit()

def show_animal_on_hand(frame, landmarks, animal_image):
    image_h, image_w, _ = frame.shape
    index_finger_tip = landmarks[8]
    cx = int(index_finger_tip.x * image_w)
    cy = int(index_finger_tip.y * image_h)

    # 이미지 크기 설정
    animal_h, animal_w = 150, 150
    animal_image_resized = cv2.resize(animal_image, (animal_w, animal_h))

    # 알파 채널 분리
    if animal_image_resized.shape[2] == 4:
        alpha_s = animal_image_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    else:
        alpha_s = np.ones((animal_h, animal_w))
        alpha_l = 0.0

    # 좌표 계산
    top_left_x = max(cx - animal_w // 2, 0)
    top_left_y = max(cy - animal_h // 2, 0)
    bottom_right_x = min(top_left_x + animal_w, frame.shape[1])
    bottom_right_y = min(top_left_y + animal_h, frame.shape[0])

    roi_width = bottom_right_x - top_left_x
    roi_height = bottom_right_y - top_left_y

    if roi_width <= 0 or roi_height <= 0:
        return frame  # 프레임 밖이면 건너뛰기

    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    animal_part = animal_image_resized[0:roi_height, 0:roi_width]
    alpha_s = alpha_s[0:roi_height, 0:roi_width]
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        roi[:, :, c] = (alpha_s * animal_part[:, :, c] + alpha_l * roi[:, :, c])

    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = roi
    return frame

# 웹캠 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if random.random() < 0.02:
                selected_animal = random.choice(list(animal_images.keys()))

            if selected_animal in animal_images:
                frame = show_animal_on_hand(frame, hand_landmarks.landmark, animal_images[selected_animal])

    cv2.imshow("Animal on Hand", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' 또는 ESC로 종료
        break

cap.release()
cv2.destroyAllWindows()


