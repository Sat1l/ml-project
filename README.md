# Meme Detector (Pose & Emotion)

ML project that recognizes human pose and emotions via webcam and matches them with a relevant meme.

## Project Structure
- `meme_detector.py` - main script for real-time detection.
- `notebooks/` - Jupyter notebooks for training and experiments.
- `src/` - preprocessing, models, and utils modules.
- `models/` - saved model weights.
- `data/` - training video data.
- `resources/memes/` - meme images.

## Installation & Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run detector:
```bash
python meme_detector.py
```

## How it works
The model uses MediaPipe Pose for body landmarks and FaceMesh for facial expressions. Extracted features are classified using SVM (Support Vector Machine) with RBF kernel.
