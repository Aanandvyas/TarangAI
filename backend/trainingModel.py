import os
import json
import av
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from safetensors.torch import load_file  # Import safetensors loader
from torch.optim.lr_scheduler import CosineAnnealingLR  # Scheduler

np.random.seed(0)

# Load pre-trained model
model_name = "facebook/timesformer-base-finetuned-k400"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForVideoClassification.from_pretrained(model_name)
pretrained_model = AutoModelForVideoClassification.from_pretrained(model_name)  # Teacher model for KD
pretrained_model.eval()  # Keep it frozen

# Load pre-trained weights
safetensors_path = r"D:\Scholar\Pro Max\Models\model.safetensors"
if os.path.exists(safetensors_path):
    print(f"Loading model weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)  # Allow partial weight loading
    print("Model weights loaded successfully!")
else:
    print(f"Error: .safetensors file not found at {safetensors_path}")
    exit(1)

# Modify classifier to accommodate both Kinetics-400 and new dance classes
num_kinetics_classes = 400  # Pre-trained classes
num_dance_classes = 5  # Bharatanatyam, Odissi, Kuchipudi, Kathak, Kathakali
num_total_classes = num_kinetics_classes + num_dance_classes

if hasattr(model, "classifier"):
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_total_classes)
elif hasattr(model, "fc"):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_total_classes)
else:
    raise AttributeError("Model does not have a 'classifier' or 'fc' layer. Check architecture.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)  # Slightly higher LR initially
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # Smoothly decay LR

# Knowledge Distillation Loss
def knowledge_distillation_loss(student_logits, teacher_logits, labels, alpha=0.3, temperature=2):
    """Combines soft loss from teacher with hard loss from true labels."""
    loss_hard = criterion(student_logits, labels)
    loss_soft = nn.KLDivLoss()(torch.log_softmax(student_logits / temperature, dim=1),
                               torch.softmax(teacher_logits / temperature, dim=1))
    return (1 - alpha) * loss_hard + alpha * loss_soft

# Sample frame extraction
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

# Dataset (Mix of original and new classes)
dataset = [
    (r"D:\Scholar\Pro Max\Projects\Datasets\Bharatnatyam\bharatnatyam_merged.mov", 400),  # Bharatanatyam
    (r"D:\Scholar\Pro Max\Projects\Datasets\Odissi\odissi_merged.mov", 401),  # Odissi
    (r"D:\Scholar\Pro Max\Projects\Datasets\Kuchipudi\kuchipudi_merged.mov", 402),  # Kuchipudi
    (r"D:\Scholar\Pro Max\Projects\Datasets\Kathak\kathak_merged.mov", 403),  # Kathak
    (r"D:\Scholar\Pro Max\Projects\Datasets\Kathakali\kathakali_merged.mov", 404),  # Kathakali
]

num_epochs = 10

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

            # Student model predictions
            outputs = model(**inputs)

            # Teacher model (pretrained) predictions (without gradients)
            with torch.no_grad():
                teacher_outputs = pretrained_model(**inputs)

            # Compute Knowledge Distillation Loss
            loss = knowledge_distillation_loss(outputs.logits, teacher_outputs.logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()  # Adjust learning rate

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}")

    # Save the trained model
    save_path = r"D:\Scholar\Pro Max\Models\timesformer_dance_finetuned"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    # Save the image processor config
    image_processor_config_path = os.path.join(save_path, "preprocessor_config.json")
    with open(image_processor_config_path, "w") as f:
        json.dump(image_processor.to_dict(), f, indent=4)

    print(f"Model and preprocessor config saved to {save_path}")
