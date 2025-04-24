import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['come', 'away', 'spin']
seq_length = 30

model = load_model('models/model2_1.0.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # ✅ 최대 2개의 손 인식
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seqs = {}  # 손마다 시퀀스를 구분해서 저장
action_seqs = {}

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for idx, res in enumerate(result.multi_hand_landmarks):
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])

            # 손 idx에 따라 다른 시퀀스 관리
            if idx not in seqs:
                seqs[idx] = []
                action_seqs[idx] = []

            seqs[idx].append(d)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seqs[idx]) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seqs[idx][-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seqs[idx].append(action)

            if len(action_seqs[idx]) < 3:
                continue

            this_action = '?'
            if action_seqs[idx][-1] == action_seqs[idx][-2] == action_seqs[idx][-3]:
                this_action = action

            cx, cy = int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0])
            cv2.putText(img, f'{this_action.upper()}', org=(cx, cy + 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
