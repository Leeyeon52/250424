import cv2
import numpy as np
import mediapipe as mp
import random
import os
import time

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
        print(f"[ERROR] Failed to load {path}")
    else:
        animal_images[name] = image
        print(f"[INFO] {name} image loaded successfully")

if not animal_images:
    print("[ERROR] No animal images loaded.")
    exit()

selected_animal = random.choice(list(animal_images.keys()))

def show_animal_on_hand(frame, landmarks, animal_image):
    image_h, image_w, _ = frame.shape
    index_finger_tip = landmarks[8]
    cx = int(index_finger_tip.x * image_w)
    cy = int(index_finger_tip.y * image_h)

    animal_h, animal_w = 150, 150
    animal_image_resized = cv2.resize(animal_image, (animal_w, animal_h), interpolation=cv2.INTER_AREA)

    if animal_image_resized.shape[2] == 4:
        alpha_s = animal_image_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    else:
        alpha_s = np.ones((animal_h, animal_w))
        alpha_l = 1.0 - alpha_s

    top_left_x = max(cx - animal_w // 2, 0)
    top_left_y = max(cy - animal_h // 2, 0)
    bottom_right_x = min(top_left_x + animal_w, frame.shape[1])
    bottom_right_y = min(top_left_y + animal_h, frame.shape[0])

    roi_width = bottom_right_x - top_left_x
    roi_height = bottom_right_y - top_left_y

    if roi_width <= 0 or roi_height <= 0:
        return frame

    animal_part = animal_image_resized[0:roi_height, 0:roi_width]
    alpha_s = alpha_s[0:roi_height, 0:roi_width]
    alpha_l = 1.0 - alpha_s

    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    for c in range(3):
        roi[:, :, c] = (alpha_s * animal_part[:, :, c] + alpha_l * roi[:, :, c])

    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = roi
    return frame

# 웹캠 실행
cap = cv2.VideoCapture(0)
prev_time = 0

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

            if selected_animal in animal_images:
                frame = show_animal_on_hand(frame, hand_landmarks.landmark, animal_images[selected_animal])

    # FPS 표시
    current_time = time.time()
    fps = 1 / (current_time - prev_time + 1e-5)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 동물 이름 표시
    cv2.putText(frame, f"Animal: {selected_animal}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

    cv2.imshow("Animal on Hand", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
    elif key == ord('n'):
        selected_animal = random.choice(list(animal_images.keys()))
        print(f"[INFO] Animal changed to: {selected_animal}")

cap.release()
cv2.destroyAllWindows()
