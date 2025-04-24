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
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[ERROR] 이미지 로딩 실패: {path}")
    else:
        animal_images[name] = image
        print(f"[INFO] {name} 이미지 로딩 완료")

if not animal_images:
    print("[FATAL] 유효한 이미지가 없습니다.")
    exit()

selected_animal = random.choice(list(animal_images.keys()))
animal_scale = 1.0  # 이미지 크기 조절 스케일

def show_animal_on_hand(frame, landmarks, animal_image):
    image_h, image_w, _ = frame.shape
    index_tip = landmarks[8]
    cx = int(index_tip.x * image_w)
    cy = int(index_tip.y * image_h)

    animal_h = int(150 * animal_scale)
    animal_w = int(150 * animal_scale)
    try:
        animal_image_resized = cv2.resize(animal_image, (animal_w, animal_h))
    except Exception as e:
        print(f"[ERROR] 이미지 리사이징 오류: {e}")
        return frame

    if animal_image_resized.shape[2] == 4:
        alpha_s = animal_image_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    else:
        alpha_s = np.ones((animal_h, animal_w))
        alpha_l = 0.0

    top_left_x = max(min(cx - animal_w // 2, image_w - animal_w), 0)
    top_left_y = max(min(cy - animal_h // 2, image_h - animal_h), 0)

    roi = frame[top_left_y:top_left_y+animal_h, top_left_x:top_left_x+animal_w]
    animal_part = animal_image_resized[0:roi.shape[0], 0:roi.shape[1]]
    alpha_s = alpha_s[0:roi.shape[0], 0:roi.shape[1]]
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        roi[:, :, c] = (alpha_s * animal_part[:, :, c] + alpha_l * roi[:, :, c])

    frame[top_left_y:top_left_y+animal_h, top_left_x:top_left_x+animal_w] = roi
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

    if key == ord('+'):
        animal_scale = min(animal_scale + 0.1, 3.0)
        print(f"[INFO] 동물 크기 증가: {animal_scale:.1f}")
    elif key == ord('-'):
        animal_scale = max(animal_scale - 0.1, 0.5)
        print(f"[INFO] 동물 크기 감소: {animal_scale:.1f}")
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
