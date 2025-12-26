import pickle
import os

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_meme_label(frame, model, extract_pose_vector_func):
    vec = extract_pose_vector_func(frame)
    if vec is None:
        return None, None

    clf = model["clf"]
    scaler = model["scaler"]
    pca = model["pca"]

    vec_scaled = scaler.transform([vec])
    vec_pca = pca.transform(vec_scaled)
    pred = clf.predict(vec_pca)[0]

    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vec_pca).max()
    
    return pred, proba
