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

def extract_pose_vector(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks.landmark
    vec = []
    for lm in landmarks:
        # vec.extend([lm.x, lm.y, lm.z]) # Обязательно добавляем lm.z
        vec.extend([lm.x, lm.y])
    vec = np.array(vec)
    vec = vec - vec.mean()
    return vec

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
