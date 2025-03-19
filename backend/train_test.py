import os
import av
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from safetensors.torch import load_file  # Import safetensors loader

np.random.seed(0)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if i > indices[-1]:
            break
    return np.stack(frames)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = clip_len * frame_sample_rate
    if converted_len >= seg_len:
        start_idx, end_idx = 0, seg_len - 1
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
    return np.linspace(start_idx, end_idx, num=clip_len, dtype=np.int64)

# Load pre-trained model and processor from Hugging Face
model_name = "facebook/timesformer-base-finetuned-k400"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForVideoClassification.from_pretrained(model_name)

# Load weights from .safetensors file
safetensors_path = r"E:\Codes\Models\model.safetensors"
if os.path.exists(safetensors_path):
    print(f"Loading model weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)  # Allow partial weight loading
    print("Model weights loaded successfully!")
else:
    print(f"Error: Could not find .safetensors file at {safetensors_path}")
    exit(1)

# Modify classifier for 2-class classification (if necessary)
num_labels = 2
if hasattr(model, "classifier"):
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
elif hasattr(model, "fc"):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)
else:
    raise AttributeError("Model does not have a 'classifier' or 'fc' layer. Check architecture.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training parameters
num_epochs = 10
dataset = [
    (r"E:\DataSets\bharatanatyam\test_main0.mov", 0),  # Bharatanatyam
    (r"E:\DataSets\odissi\test_main1.mov", 1),  # Odissi
]

if __name__ == "__main__":
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for file_path, label in dataset:
            if not os.path.exists(file_path):
                print(f"Warning: Dataset path '{file_path}' not found. Skipping...")
                continue
            
            try:
                container = av.open(file_path)
                indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
                video = read_video_pyav(container, indices)
            except Exception as e:
                print(f"Error reading video '{file_path}': {e}")
                continue

            # Convert video frames to tensor
            inputs = image_processor(list(video), return_tensors="pt").to(device)
            labels = torch.tensor([label], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Save the trained model
    save_path = r"E:\Codes\Models\timesformer_bharatanatyam_odissi"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    print(f"Model saved to {save_path}")
