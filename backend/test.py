import os
import shutil
import pandas as pd

# Define dataset path
dataset_path = r"E:\DataSets\tarangDataset - Copy"
output_metadata_file = os.path.join(dataset_path, "metadata.csv")

# Ensure consistent structure across dance forms
dance_forms = ["bharatnatyam", "kathakali", "sattriya"]
splits = ["Training", "Valid", "Test"]

for dance in dance_forms:
    dance_path = os.path.join(dataset_path, dance)
    if not os.path.exists(dance_path):
        continue  # Skip missing folders
    
    # Create Training, Valid, Test directories if missing
    for split in splits:
        split_path = os.path.join(dance_path, split)
        os.makedirs(split_path, exist_ok=True)

# Collect metadata
metadata = []

for dance in dance_forms:
    for split in splits:
        split_path = os.path.join(dataset_path, dance, split)
        if not os.path.exists(split_path):
            continue
        
        for mudra in os.listdir(split_path):
            mudra_path = os.path.join(split_path, mudra)
            if not os.path.isdir(mudra_path):
                continue
            
            for file in os.listdir(mudra_path):
                file_path = os.path.join(mudra_path, file)
                
                # Only consider images/videos
                if file.lower().endswith((".jpg", ".png", ".mp4", ".avi")):
                    openpose_json = file.replace(".jpg", ".json").replace(".png", ".json").replace(".mp4", ".json").replace(".avi", ".json")
                    openpose_path = os.path.join(mudra_path, openpose_json)
                    has_openpose = os.path.exists(openpose_path)
                    
                    metadata.append([dance, split, mudra, file, has_openpose])

# Save metadata as CSV
metadata_df = pd.DataFrame(metadata, columns=["Dance Form", "Split", "Mudra", "File", "Has OpenPose"])
metadata_df.to_csv(output_metadata_file, index=False)

print(f"Dataset organized! Metadata saved to {output_metadata_file}")
