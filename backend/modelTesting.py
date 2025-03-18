import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch.nn.functional as F

np.random.seed(0)

def read_video_pyav(container, indices):
    """
    Decode the video with PyAV and extract specific frames.
    """
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if i > indices[-1]:  # Stop decoding early
            break
    return np.stack(frames)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from a video.
    """
    converted_len = clip_len * frame_sample_rate
    if converted_len >= seg_len:  # Handle short videos
        start_idx, end_idx = 0, seg_len - 1
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len

    return np.linspace(start_idx, end_idx, num=clip_len, dtype=np.int64)

# 🔹 Change this to the path of your local video file
file_path = r"C:\Users\chari\Downloads\Timeline 1.mov"

# Load local video file
container = av.open(file_path)

# Sample frames
indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)

# Load Pretrained Video Classification Model
model_name = "facebook/timesformer-base-finetuned-k400"  # Model trained on Kinetics-400
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForVideoClassification.from_pretrained(model_name)

# Preprocess Video and Get Predictions
inputs = image_processor(list(video), return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits  # Raw scores before softmax
probs = F.softmax(logits, dim=-1)  # Convert to probabilities
predicted_class = torch.argmax(probs, dim=-1).item()  # Get the class index

# Load Kinetics-400 Labels
id2label = model.config.id2label  # Mapping of class index to label
predicted_label = id2label[predicted_class]

# Print the Predicted Action Label
print(f"Predicted Action: {predicted_label}")
