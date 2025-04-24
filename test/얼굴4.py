import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Mediapipe 얼굴 랜드마크 세팅 ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# === 2. 정규화 함수 (눈 사이 거리 기준) ===
def normalize_landmarks(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    return (landmarks - left_eye) / eye_dist

# === 3. 이미지에서 랜드마크 추출 ===
def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ 이미지 {image_path} 로드 실패!")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"❌ 얼굴 인식 실패: {image_path}")
        return None

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = []
    h, w, _ = image.shape
    for lm in face_landmarks.landmark:
        landmarks.append([lm.x * w, lm.y * h])

    return np.array(landmarks)

# === 4. 모든 등록 이미지 벡터 저장 ===
def load_reference_vectors(folder_path):
    ref_vectors = {}
    for file in os.listdir(folder_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(folder_path, file)
            lm = extract_landmarks_from_image(path)
            if lm is not None:
                norm = normalize_landmarks(lm).flatten()
                ref_vectors[file] = norm
    return ref_vectors

# === 5. 비교 이미지와 유사도 측정 ===
def compare_with_references(test_image_path, ref_vectors, threshold=0.95):
    lm = extract_landmarks_from_image(test_image_path)
    if lm is None:
        return "❌ 얼굴 인식 실패"

    norm = normalize_landmarks(lm).flatten()

    for name, ref_vec in ref_vectors.items():
        sim = cosine_similarity([norm], [ref_vec])[0][0]
        print(f"🔍 {name} 와 유사도: {sim:.4f}")
        if sim > threshold:
            return f"✅ {name} 과 일치 (유사도: {sim:.4f})"
    return "❌ 일치하는 얼굴 없음"

# === 6. 실행 ===
if __name__ == "__main__":
    # 등록된 벡터 불러오기
    ref_vectors = load_reference_vectors("snapshots")

    # 테스트할 이미지 지정
    test_image = "snapshots/face_snapshot_1745463983.png"  # 예시

    result = compare_with_references(test_image, ref_vectors)
    print(result)
