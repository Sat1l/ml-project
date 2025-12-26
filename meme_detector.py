import cv2
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import extract_pose_vector
from models import load_model, predict_meme_label
from utils import load_memes

MEME_IMAGES = {
    str(i): f"./resources/memes/{i}.jpg" for i in range(1, 9)
}

MODEL_PATH = "./models/model.pkl"
CONFIDENCE_THRESHOLD = 0.7

def main():
    loaded_memes_dict = load_memes(MEME_IMAGES)
    
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: camera not found")
        return

    print("Camera opened. Starting meme detector...")
    print(f"(Confidence threshold: {CONFIDENCE_THRESHOLD})")
    print("(Press 'Q' to exit)")
    
    current_meme_cls = None
    current_meme_img = None
    current_proba = None
    show_until = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        pred_cls, proba = predict_meme_label(frame, model, extract_pose_vector)
        
        if pred_cls is not None and proba is not None and proba >= CONFIDENCE_THRESHOLD and pred_cls in loaded_memes_dict:
            current_meme_cls = pred_cls
            current_proba = proba
            meme_img = loaded_memes_dict[pred_cls].copy()
    
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
            current_proba = None
    
        if current_meme_cls is not None and current_proba is not None:
            cv2.putText(frame, f"MEME: {current_meme_cls} ({current_proba:.0%})", (30, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        cv2.imshow("Live Meme Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    main()
