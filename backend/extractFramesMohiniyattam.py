import cv2
import os
import pandas as pd
from pathlib import Path
import time

def extract_frames(source_dir, target_dir, fps_extract=3, target_size=(512, 512)):
    os.makedirs(target_dir, exist_ok=True)
    metadata_file = os.path.join(os.path.dirname(target_dir), "metadata.csv")
    metadata = []
    
    video_files = [f for f in os.listdir(source_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"Found {len(video_files)} video files to process")
    
    total_frames_extracted = 0
    
    for video_file in video_files:
        video_path = os.path.join(source_dir, video_file)
        print(f"Processing {video_file}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            continue
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps * fps_extract)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                resized_frame = cv2.resize(frame, target_size)
                
                base_name = os.path.splitext(video_file)[0]
                output_filename = f"{base_name}_frame_{saved_count:04d}.jpg"
                output_path = os.path.join(target_dir, output_filename)
                
                cv2.imwrite(output_path, resized_frame)
                
                timestamp = frame_count / video_fps
                
                metadata.append({
                    'video_filename': video_file,
                    'frame_number': saved_count,
                    'timestamp_seconds': timestamp,
                    'image_path': output_path,
                    'width': target_size[0],
                    'height': target_size[1],
                    'dance_form': 'mohiniyattam'
                })
                
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        total_frames_extracted += saved_count
        print(f"Extracted {saved_count} frames from {video_file}")
    
    metadata_df = pd.DataFrame(metadata)
    
    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file)
        updated_df = pd.concat([existing_df, metadata_df], ignore_index=True)
        updated_df.to_csv(metadata_file, index=False)
        print(f"Updated metadata file: {metadata_file}")
    else:
        metadata_df.to_csv(metadata_file, index=False)
        print(f"Created new metadata file: {metadata_file}")
    
    print(f"Total frames extracted: {total_frames_extracted}")
    print(f"Total videos processed: {len(video_files)}")

if __name__ == "__main__":
    source_dir = r"E:\DataSets\tarangDataset(main) step2\odissi"
    target_dir = r"E:\DataSets\TarangAI_Dataset\images\odissi"
    
    start_time = time.time()
    extract_frames(source_dir, target_dir, fps_extract=3)
    elapsed_time = time.time() - start_time
    
    print(f"Processing completed in {elapsed_time:.2f} seconds")