'''needs to be updated to fine tuning code'''
import os
import json
import av
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from safetensors.torch import load_file
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

np.random.seed(0)

# this is to load timesformer model
model_name = "facebook/timesformer-base-finetuned-k400"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForVideoClassification.from_pretrained(model_name)
pretrained_model = AutoModelForVideoClassification.from_pretrained(model_name)  # Teacher model for KD(Knowledge Distillation)
pretrained_model.eval() 

# Loading pre-trained weights
safetensors_path = r"D:\Scholar\Pro Max\Models\model.safetensors"
if os.path.exists(safetensors_path):
    print(f"Loading model weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)  # Allow partial weight loading
    print("Model weights loaded successfully!")
else:
    print(f"Error: .safetensors file not found at {safetensors_path}")
    exit(1)

# Modify classifier to accommodate both Kinetics-400 along with new dance classes(Bharatnatyam,Odissi,Kuchipudi,Kathak,Kathakali)
num_kinetics_classes = 400  
num_dance_classes = 5  
num_total_classes = num_kinetics_classes + num_dance_classes

if hasattr(model, "classifier"):
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),  # Dropout to prevent overfitting
        nn.Linear(in_features, num_total_classes)
    )
elif hasattr(model, "fc"):
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Dropout to prevent overfitting
        nn.Linear(in_features, num_total_classes)
    )
else:
    raise AttributeError("Model does not have a 'classifier' or 'fc' layer.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

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

# Data Augmentation: Applies little noise to frames
def augment_frames(frames):
    """Applies slight noise to frames to prevent overfitting."""
    noise = np.random.normal(0, 5, frames.shape).astype(np.uint8)
    return np.clip(frames + noise, 0, 255)


dataset = [
    (r"D:\Scholar\Pro Max\Projects\Datasets\Bharatnatyam\bharatnatyam_merged.mov", 400),
    (r"D:\Scholar\Pro Max\Projects\Datasets\Odissi\odissi_merged.mov", 401),
    (r"D:\Scholar\Pro Max\Projects\Datasets\Kuchipudi\kuchipudi_merged.mov", 402),
    (r"D:\Scholar\Pro Max\Projects\Datasets\Kathak\kathak_merged.mov", 403),
    (r"D:\Scholar\Pro Max\Projects\Datasets\Kathakali\kathakali_merged.mov", 404),
]

# Splitting data into 80% training and 20% validation for early stopping
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

num_epochs = 10
best_val_loss = float("inf")
patience = 3  # Stop training if validation loss does not improve for 3 epochs
patience_counter = 0

if __name__ == "__main__":
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for file_path, label in train_data:
            if not os.path.exists(file_path):
                print(f"Warning: Dataset path '{file_path}' not found. Skipping...")
                continue

            try:
                container = av.open(file_path)
                indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
                video = read_video_pyav(container, indices)
                video = augment_frames(video)  # Apply augmentation
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

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for file_path, label in val_data:
                if not os.path.exists(file_path):
                    continue

                try:
                    container = av.open(file_path)
                    indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
                    video = read_video_pyav(container, indices)
                except Exception as e:
                    continue

                inputs = image_processor(list(video), return_tensors="pt").to(device)
                labels = torch.tensor([label], dtype=torch.long).to(device)

                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save the trained model
    save_path = r"D:\Scholar\Pro Max\Models\timesformer_dance_finetuned"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    # Save the image processor config
    with open(os.path.join(save_path, "preprocessor_config.json"), "w") as f:
        json.dump(image_processor.to_dict(), f, indent=4)

    print(f"Model and preprocessor config saved to {save_path}")
