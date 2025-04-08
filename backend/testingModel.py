import os
import av
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForVideoClassification

# while running check if it is calling PREPROCESSOR_CONFIG.json file (IMP)

# Path to test video
test_video_path = r"C:\Users\chari\Downloads\bharatnatyamtest3.mp4"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model path (where your trained model is saved)
model_path = r"D:\Scholar\Pro Max\Models\timesformer_dance_finetuned"

# Load the processor and model
image_processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForVideoClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Function to read video frames
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if i > indices[-1]:
            break
    return np.stack(frames) if frames else None

# Function to sample frames from the video
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = clip_len * frame_sample_rate
    if converted_len >= seg_len:
        start_idx, end_idx = 0, seg_len - 1
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
    return np.linspace(start_idx, end_idx, num=clip_len, dtype=np.int64)

# Define label mapping (Must match training labels)
num_kinetics_classes = 400  # Existing Kinetics-400 classes
dance_labels = {400: "Bharatanatyam", 401: "Odissi", 402: "Kuchipudi", 403: "Kathak", 404: "Kathakali"}
all_labels = {**{i: f"Kinetics-{i}" for i in range(num_kinetics_classes)}, **dance_labels}


# Check if file exists
if not os.path.exists(test_video_path):
    raise FileNotFoundError(f"Test video '{test_video_path}' not found.")

# Read video and process it
try:
    container = av.open(test_video_path)
    total_frames = container.streams.video[0].frames
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=total_frames)
    video = read_video_pyav(container, indices)

    if video is None:
        raise RuntimeError("Failed to extract frames. Check video format or frame extraction logic.")
except Exception as e:
    raise RuntimeError(f"Error reading video: {e}")

# Convert video frames to tensor
inputs = image_processor(list(video), return_tensors="pt").to(device)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Get the predicted label
predicted_label = all_labels.get(predicted_class, "Unknown")

print(f"Predicted Dance Form: {predicted_label}")