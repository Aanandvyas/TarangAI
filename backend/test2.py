import cv2
import os

video_path = r"E:\DataSets\tarangDataset - Copy\bharatnatyam\videos\DhalankuAdavu1.mp4"
output_folder = r"E:\DataSets\tarangDataset - Copy\bharatnatyam\Training"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 1  # Extract one frame per second

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
        frame_filename = os.path.join(output_folder, f"frame_{frame_id}.jpg")
        cv2.imwrite(frame_filename, frame)
    frame_id += 1

cap.release()
print(f"Frames saved in {output_folder}")