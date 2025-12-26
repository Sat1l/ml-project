import cv2
import time
import os
import mediapipe as mp
import numpy as np
import pickle

# Маппинг: класс (строка, как в y) → путь к картинке мема
MEME_IMAGES = {
    "1": "./memes/1.jpg",
    "2": "./memes/2.jpg",
    "3": "./memes/3.jpg",
    "4": "./memes/4.jpg",
    "5": "./memes/5.jpg",
    "6": "./memes/6.jpg",
    "7": "./memes/7.jpg",
    "8": "./memes/8.jpg",
}

# Загружаем все мемы заранее
loaded_memes = {}
for cls, path in MEME_IMAGES.items():
    if os.path.exists(path):
        img = cv2.imread(path)
        loaded_memes[cls] = img
    else:
        print("⚠️ Не найден файл мема для класса", cls, "по пути", path)

# Инициализируем MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Инициализируем FaceMesh для эмоций
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_face_features(rgb_frame):
    """
    Извлекает признаки эмоций из лица используя FaceMesh.
    """
    result = face_mesh.process(rgb_frame)
    N_FACE_FEATURES = 20
    
    if not result.multi_face_landmarks:
        return np.zeros(N_FACE_FEATURES)
    
    face_landmarks = result.multi_face_landmarks[0].landmark
    nose = face_landmarks[1]
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    eye_dist = max(np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2), 0.01)
    
    features = []
    # Открытость глаз
    features.extend([(face_landmarks[159].y - face_landmarks[145].y) / eye_dist,
                     (face_landmarks[386].y - face_landmarks[374].y) / eye_dist])
    # Положение бровей
    features.extend([(face_landmarks[105].y - face_landmarks[159].y) / eye_dist,
                     (face_landmarks[334].y - face_landmarks[386].y) / eye_dist])
    # Рот
    mouth_open = (face_landmarks[14].y - face_landmarks[13].y) / eye_dist
    mouth_width = (face_landmarks[308].x - face_landmarks[78].x) / eye_dist
    features.extend([mouth_open, mouth_width])
    mouth_center_y = (face_landmarks[13].y + face_landmarks[14].y) / 2
    features.extend([(mouth_center_y - face_landmarks[78].y) / eye_dist,
                     (mouth_center_y - face_landmarks[308].y) / eye_dist])
    # Голова
    features.extend([(left_eye.y - right_eye.y) / eye_dist,
                     (nose.x - (left_eye.x + right_eye.x) / 2) / eye_dist])
    features.append((face_landmarks[13].y - nose.y) / eye_dist)
    for idx in [33, 263, 61, 291, 199, 175, 152, 10, 234]:
        features.append((face_landmarks[idx].x - nose.x) / eye_dist)
    
    return np.array(features[:N_FACE_FEATURES])


def extract_pose_vector(frame):
    """
    Комбинирует позу тела, признаки лица и явные признаки рук.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks.landmark
    
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    
    center_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x + 
                landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 4
    center_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y + 
                landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 4
    
    shoulder_dist = max(np.sqrt(
        (landmarks[LEFT_SHOULDER].x - landmarks[RIGHT_SHOULDER].x) ** 2 +
        (landmarks[LEFT_SHOULDER].y - landmarks[RIGHT_SHOULDER].y) ** 2
    ), 0.01)

    vec = []
    for lm in landmarks:
        norm_x = (lm.x - center_x) / shoulder_dist
        norm_y = (lm.y - center_y) / shoulder_dist
        vec.extend([norm_x, norm_y, lm.visibility])

    # Явные признаки видимости и положения рук
    left_wrist_vis = landmarks[LEFT_WRIST].visibility
    right_wrist_vis = landmarks[RIGHT_WRIST].visibility
    left_hand_visible = 1.0 if left_wrist_vis > 0.5 else 0.0
    right_hand_visible = 1.0 if right_wrist_vis > 0.5 else 0.0
    any_hand_visible = 1.0 if (left_wrist_vis > 0.5 or right_wrist_vis > 0.5) else 0.0
    left_wrist_above_shoulder = 1.0 if landmarks[LEFT_WRIST].y < landmarks[LEFT_SHOULDER].y else 0.0
    right_wrist_above_shoulder = 1.0 if landmarks[RIGHT_WRIST].y < landmarks[RIGHT_SHOULDER].y else 0.0
    left_hand_dist = np.sqrt((landmarks[LEFT_WRIST].x - center_x)**2 + 
                             (landmarks[LEFT_WRIST].y - center_y)**2) / shoulder_dist
    right_hand_dist = np.sqrt((landmarks[RIGHT_WRIST].x - center_x)**2 + 
                              (landmarks[RIGHT_WRIST].y - center_y)**2) / shoulder_dist
    
    vec.extend([left_hand_visible, right_hand_visible, any_hand_visible,
                left_wrist_above_shoulder, right_wrist_above_shoulder,
                left_hand_dist, right_hand_dist])

    # Признаки лица
    face_features = extract_face_features(rgb)
    vec.extend(face_features)

    return np.array(vec)

# Загружаем модель
with open("./data/model.pkl", "rb") as f:
    model = pickle.load(f)

clf = model["clf"]
scaler = model["scaler"]
pca = model["pca"]
classes = model["classes"]

def predict_meme_label(frame):
    vec = extract_pose_vector(frame)
    if vec is None:
        return None, None

    vec_scaled = scaler.transform([vec])
    vec_pca = pca.transform(vec_scaled)
    pred = clf.predict(vec_pca)[0]

    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vec_pca).max()
    return pred, proba

# Основной цикл приложения
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Ошибка: не удается открыть вебкамеру")
else:
    print("✅ Вебкамера открыта. Запуск детектора мемов...")
    print("(Нажмите 'Q' чтобы выйти)")
    
    current_meme_cls = None
    current_meme_img = None
    show_until = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        pred_cls, proba = predict_meme_label(frame)
        if pred_cls is not None and pred_cls in loaded_memes:
            current_meme_cls = pred_cls
            meme_img = loaded_memes[pred_cls].copy()
    
            h, w, _ = frame.shape
            mw = w // 3
            mh = int(mw * meme_img.shape[0] / meme_img.shape[1])
            meme_img = cv2.resize(meme_img, (mw, mh))
    
            current_meme_img = meme_img
            show_until = time.time() + 1.0
    
        if current_meme_img is not None and time.time() < show_until:
            mh, mw, _ = current_meme_img.shape
            x0, y0 = 10, 10
            x1, y1 = x0 + mw, y0 + mh
            if y1 <= frame.shape[0] and x1 <= frame.shape[1]:
                frame[y0:y1, x0:x1] = current_meme_img
        else:
            current_meme_img = None
            current_meme_cls = None
    
        if current_meme_cls is not None:
            cv2.putText(frame, f"MEME: {current_meme_cls}", (30, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        cv2.imshow("Live Meme Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Приложение закрыто")
