import cv2
import numpy as np
import mediapipe as mp
import random
import os

# 1. Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 2. 동물 이미지 로딩
animal_images = {}
animal_names = ['dog', 'cat', 'panda', 'fox', 'dino']
for name in animal_names:
    path = f'images/{name}.png'
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함
    if image is None:
        print(f"[WARNING] {name}.png 이미지가 없습니다.")
    elif image.shape[2] != 4:
        print(f"[WARNING] {name}.png 에 알파 채널이 없습니다.")
    else:
        animal_images[name] = image
        print(f"[INFO] {name} 이미지 로딩 성공.")

if not animal_images:
    print("[ERROR] 사용할 수 있는 동물 이미지가 없습니다. 프로그램 종료.")
    exit()

# 3. 초기값 설정
selected_animal = random.choice(list(animal_images.keys()))
animal_scale = 1.0  # 초기 동물 크기 비율

def show_animal_on_hand(frame, landmarks, animal_image, scale=1.0):
    image_h, image_w, _ = frame.shape
    index_finger_tip = landmarks[8]
    cx = int(index_finger_tip.x * image_w)
    cy = int(index_finger_tip.y * image_h)

    # 동물 이미지 크기 설정
    animal_h, animal_w = int(150 * scale), int(150 * scale)
    animal_image_resized = cv2.resize(animal_image, (animal_w, animal_h), interpolation=cv2.INTER_AREA)

    # 알파 채널 분리
    if animal_image_resized.shape[2] == 4:
        alpha_s = animal_image_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    else:
        print("[ERROR] 알파 채널이 없습니다.")
        return frame

    # 위치 계산
    top_left_x = max(cx - animal_w // 2, 0)
    top_left_y = max(cy - animal_h // 2, 0)
    bottom_right_x = min(top_left_x + animal_w, frame.shape[1])
    bottom_right_y = min(top_left_y + animal_h, frame.shape[0])

    roi_width = bottom_right_x - top_left_x
    roi_height = bottom_right_y - top_left_y

    if roi_width <= 0 or roi_height <= 0:
        return frame  # 프레임 밖이면 무시

    # ROI 적용
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    animal_part = animal_image_resized[0:roi_height, 0:roi_width]
    alpha_s = alpha_s[0:roi_height, 0:roi_width]
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        roi[:, :, c] = (alpha_s * animal_part[:, :, c] + alpha_l * roi[:, :, c]).astype(np.uint8)

    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = roi
    return frame

# 4. 웹캠 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[WARNING] 카메라 프레임을 읽을 수 없습니다.")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 동물 무작위 변경 (낮은 확률)
            if random.random() < 0.02:
                selected_animal = random.choice(list(animal_images.keys()))

            # 동물 이미지 표시
            frame = show_animal_on_hand(frame, hand_landmarks.landmark, animal_images[selected_animal], animal_scale)

    # UI 출력
    cv2.putText(frame, f"Animal: {selected_animal}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Scale: {animal_scale:.1f}  [+/-로 크기조절]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    cv2.putText(frame, "Press [q] or [ESC] to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)

    cv2.imshow("Animal on Hand", frame)

    key = cv2.waitKey(1)

    # 동물 크기 조절
    if key == ord('+'):
        animal_scale = min(animal_scale + 0.1, 3.0)
        print(f"[INFO] 동물 크기 증가: {animal_scale:.1f}")
    elif key == ord('-'):
        animal_scale = max(animal_scale - 0.1, 0.5)
        print(f"[INFO] 동물 크기 감소: {animal_scale:.1f}")

    # 프로그램 종료
    elif key == ord('q') or key == 27:
        print("[INFO] 프로그램 종료")
        break

cap.release()
cv2.destroyAllWindows()
