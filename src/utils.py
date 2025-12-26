import cv2
import os

def load_memes(meme_images_mapping):
    loaded_memes = {}
    for cls, path in meme_images_mapping.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            loaded_memes[cls] = img
        else:
            print(f"Meme file not found for class {cls} at {path}")
    return loaded_memes
