import os
import cv2
import pandas as pd

# Paths and Config
DATASET_PATH = r"E:\DataSets\tarangDataset\bharatnatyam"
EXCEL_FILE = os.path.join(DATASET_PATH, "adavus_mapping.xlsx")  # Update to your XLSX name
FRAME_RATE = 1  # 1 frame per second

def extract_frames(video_path, output_folder, frame_rate=1):
    """Extract frames from a video at ~frame_rate frames per second."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback if FPS not detected

    # Interval in frames between extractions
    frame_interval = max(1, int(fps // frame_rate))

    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        frame_count += 1

    cap.release()

def main():
    # 1. Read the Excel file with columns "video_name" and "adavu_name"
    df = pd.read_excel(EXCEL_FILE)

    # 2. For each row, extract frames to "extracted/[adavu_name]/[video_name_without_ext]"
    for idx, row in df.iterrows():
        video_name = str(row["video_name"]).strip()
        adavu_name = str(row["adavu_name"]).strip()

        # Skip empty lines or missing video names
        if not video_name or not adavu_name or video_name == "nan":
            print(f"Skipping row {idx}: missing data.")
            continue

        video_full_path = os.path.join(DATASET_PATH, video_name)
        if not os.path.exists(video_full_path):
            print(f"❌ Video not found: {video_full_path}")
            continue

        # Example: videos/Thattadavu1.mp4 -> "Thattadavu1"
        video_base = os.path.splitext(os.path.basename(video_name))[0]

        # Output folder => E:\DataSets\tarangDataset\bharatnatyam\extracted\Thattadavu\Thattadavu1
        output_folder = os.path.join(DATASET_PATH, "extracted", adavu_name, video_base)

        print(f"Extracting frames from {video_full_path} → {output_folder}")
        extract_frames(video_full_path, output_folder, FRAME_RATE)

if __name__ == "__main__":
    main()
    print("✅ Frame extraction complete!")
