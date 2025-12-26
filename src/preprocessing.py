import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_face_features(rgb_frame):
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
    # Eyes openness
    features.extend([(face_landmarks[159].y - face_landmarks[145].y) / eye_dist,
                     (face_landmarks[386].y - face_landmarks[374].y) / eye_dist])
    # Brows position
    features.extend([(face_landmarks[105].y - face_landmarks[159].y) / eye_dist,
                     (face_landmarks[334].y - face_landmarks[386].y) / eye_dist])
    # Mouth state
    mouth_open = (face_landmarks[14].y - face_landmarks[13].y) / eye_dist
    mouth_width = (face_landmarks[308].x - face_landmarks[78].x) / eye_dist
    features.extend([mouth_open, mouth_width])
    mouth_center_y = (face_landmarks[13].y + face_landmarks[14].y) / 2
    features.extend([(mouth_center_y - face_landmarks[78].y) / eye_dist,
                     (mouth_center_y - face_landmarks[308].y) / eye_dist])
    # Head pose
    features.extend([(left_eye.y - right_eye.y) / eye_dist,
                     (nose.x - (left_eye.x + right_eye.x) / 2) / eye_dist])
    features.append((face_landmarks[13].y - nose.y) / eye_dist)
    for idx in [33, 263, 61, 291, 199, 175, 152, 10, 234]:
        features.append((face_landmarks[idx].x - nose.x) / eye_dist)
    
    return np.array(features[:N_FACE_FEATURES])

def extract_pose_vector(frame):
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

    # Hand visibility and position
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

    # Face features
    face_features = extract_face_features(rgb)
    vec.extend(face_features)

    return np.array(vec)
