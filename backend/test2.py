#!THIS CODE WILL GIVE YOU THE FRAME's FROM THE VIDEOS IN THE DATASET
import os
import cv2
import pandas as pd

#!EDIT BEFORE RUNNING THE CODE
DATASET_PATH = r"E:\DataSets\tarangDataset\odissi"
EXCEL_FILENAME = "odissi_videos.xlsx"
VIDEO_FOLDER = "videos"
FRAME_RATE = 1

def extract_frames(video_path, output_folder, frame_rate=1):
    """Extract frames from a video at approximately `frame_rate` frames per second."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    #Below condition is not very much important
    if fps <= 0:
        fps = 30

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
    excel_path = os.path.join(DATASET_PATH, EXCEL_FILENAME)
    if not os.path.exists(excel_path):
        print(f"❌ Excel file not found: {excel_path}")
        return

    df = pd.read_excel(excel_path)

    required_cols = {"video_name", "video_category"}
    if not required_cols.issubset(df.columns):
        print(f"X The Excel file must contain columns: {required_cols}")
        return

    for idx, row in df.iterrows():
        video_name = str(row["video_name"]).strip()
        video_category = str(row["video_category"]).strip()

        video_full_path = os.path.join(DATASET_PATH, VIDEO_FOLDER, video_name)
        if not os.path.exists(video_full_path):
            print(f"❌ Video file not found (row {idx}): {video_full_path}")
            continue

        category_clean = video_category.replace(" ", "_")

        video_base = os.path.splitext(video_name)[0]
        video_base_clean = video_base.replace(" ", "_").replace(".", "_")

        output_folder = os.path.join(
            DATASET_PATH,
            "extracted",
            category_clean,
            video_base_clean
        )

        print(f"[{idx}] Extracting frames from: {video_full_path}")
        print(f" → {output_folder}")
        extract_frames(video_full_path, output_folder, FRAME_RATE)

if __name__ == "__main__":
    main()
    print(" Frame extraction complete!")
