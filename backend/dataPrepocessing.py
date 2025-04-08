import os
import cv2
import json
import mediapipe as mp
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import albumentations as A

# Set image size
IMAGE_SIZE = 256

# Initialize MediaPipe Pose and Face Detection models
mp_pose = mp.solutions.pose.Pose()
mp_hands = mp.solutions.hands.Hands()
mp_face = mp.solutions.face_mesh.FaceMesh()

# Augmentation Pipeline (Rotation, Flip, Brightness Adjustment)
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5)
])

# Create dataset folders if they don't exist
DATASET_DIR = "Dataset"
OUTPUT_CSV = os.path.join(DATASET_DIR, "metadata.csv")
KEYPOINTS_DIR = os.path.join(DATASET_DIR, "keypoints")
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

# Helper function to extract MediaPipe keypoints
def extract_keypoints(image):
    """Extracts pose, hand, and face keypoints using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = {"pose": [], "hands": [], "face": []}

    # Pose detection
    results_pose = mp_pose.process(image_rgb)
    if results_pose.pose_landmarks:
        keypoints["pose"] = [[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]

    # Hand detection
    results_hands = mp_hands.process(image_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            keypoints["hands"].append([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Face detection
    results_face = mp_face.process(image_rgb)
    if results_face.multi_face_landmarks:
        keypoints["face"] = [[lm.x, lm.y, lm.z] for lm in results_face.multi_face_landmarks[0].landmark]

    return keypoints

# Process videos: Convert to frames
def process_video(video_path, output_folder):
    """Extracts frames from videos and saves them as images."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame_resized)
        frames_list.append(frame_path)
        frame_count += 1

    cap.release()
    return frames_list

# Process images: Resize, augment, extract keypoints
def process_images(image_path, output_folder):
    """Resizes, augments, and extracts keypoints from images."""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    new_image_path = os.path.join(output_folder, f"{base_name}.jpg")
    cv2.imwrite(new_image_path, image_resized)

    # Extract keypoints
    keypoints = extract_keypoints(image_resized)

    # Save keypoints to JSON
    json_path = os.path.join(KEYPOINTS_DIR, f"{base_name}.json")
    with open(json_path, "w") as json_file:
        json.dump(keypoints, json_file, indent=4)

    # Perform Augmentation
    augmented_paths = []
    for i in range(2):  # Create 2 augmented versions per image
        augmented = augmentation(image=image_resized)["image"]
        aug_path = os.path.join(output_folder, f"{base_name}_aug{i+1}.jpg")
        cv2.imwrite(aug_path, augmented)
        augmented_paths.append(aug_path)

    return new_image_path, augmented_paths, json_path

# Process dataset and generate CSV
metadata = []
dance_forms = ["Bharatanatyam", "Kathak", "Kuchipudi", "Kathakali", "Odissi", "Sattriya"]
clothing_labels = {
    "Bharatanatyam": "Saree",
    "Kathak": "Anarkali Dress",
    "Kuchipudi": "Saree",
    "Kathakali": "Traditional Costume",
    "Odissi": "Sambalpuri Saree",
    "Sattriya": "Mekhela Chador"
}

for dance_form in tqdm(dance_forms, desc="Processing Dance Forms"):
    image_folder = os.path.join(DATASET_DIR, dance_form, "images")
    video_folder = os.path.join(DATASET_DIR, dance_form, "vids")

    for video_file in glob(os.path.join(video_folder, "*.mp4")):
        frames = process_video(video_file, image_folder)

        for frame in frames:
            img_path, aug_paths, keypoints_path = process_images(frame, image_folder)
            metadata.append([os.path.basename(frame), dance_form, os.path.basename(video_file), img_path, keypoints_path, clothing_labels[dance_form]])
            
            for aug_img in aug_paths:
                metadata.append([os.path.basename(aug_img), dance_form, os.path.basename(video_file), aug_img, keypoints_path, clothing_labels[dance_form]])

    for image_file in glob(os.path.join(image_folder, "*.jpg")):
        img_path, aug_paths, keypoints_path = process_images(image_file, image_folder)
        metadata.append([os.path.basename(image_file), dance_form, "N/A", img_path, keypoints_path, clothing_labels[dance_form]])
        
        for aug_img in aug_paths:
            metadata.append([os.path.basename(aug_img), dance_form, "N/A", aug_img, keypoints_path, clothing_labels[dance_form]])

# Save metadata to CSV
df = pd.DataFrame(metadata, columns=["id", "dance_form", "video_file", "image_file", "keypoints_file", "clothing"])
df.to_csv(OUTPUT_CSV, index=False)

print("Preprocessing complete. Metadata saved to", OUTPUT_CSV)