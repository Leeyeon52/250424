import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 이미지가 저장된 폴더 경로
image_folder = r"C:\Users\302-15\Desktop\test\snapshots"

# 해당 폴더에서 모든 PNG, JPG 파일을 가져오기
allowed_images = [
    os.path.join(image_folder, filename)
    for filename in os.listdir(image_folder)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg'))  # PNG, JPG, JPEG 파일만 필터링
]

# 디버깅용: 가져온 이미지 파일 경로 출력
print("허용된 이미지 파일들:")
for img_path in allowed_images:
    print(img_path)

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# 기준 얼굴 좌표 추출
reference_faces = []
for path in allowed_images:
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ 이미지 {path} 로드 실패!")
        continue
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    if results.multi_face_landmarks:
        landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
        reference_faces.append(landmarks)
    else:
        print(f"❌ 얼굴 인식 실패: {path}")

# 실시간 처리용 Mediapipe 다시 설정
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# 실시간 인식 시작
cap = cv2.VideoCapture(0)
prev_time = time.time()
diff_count = 0
MAX_DIFF_COUNT = 35

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

        matched = False
        for ref in reference_faces:
            if ref.shape == current_landmarks.shape:
                diff = np.linalg.norm(ref - current_landmarks)
                if diff < 0.04:
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

        # 시각화
        points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark], dtype=np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

        for (x, y) in points:
            cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

        mp_drawing.draw_landmarks(
            image=overlay,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
        )

    # 배경 어둡게 처리
    mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_blur = cv2.merge([mask_blur] * 3) / 255.0
    result = (overlay.astype(np.float32) * mask_blur + dark_bg.astype(np.float32) * (1 - mask_blur)).astype(np.uint8)

    # FPS 출력
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(result, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Face Image Auth Check', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
