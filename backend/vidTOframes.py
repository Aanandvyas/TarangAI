import cv2
import os

def extract_frames(video_path, output_folder, frame_count=16):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_gap = total_frames // frame_count
    
    frames = []
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_gap)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frames.append(frame)

        frame_filename = os.path.join(output_folder, f"frame_{i+1}.jpg")
        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cap.release()
    return frames

video_path = r"D:\Downloads\test_breakDance.mp4"
output_folder = r"D:\Scholar\Bhaskar Chari 2022BAI10155\Extracted Frames"

frames = extract_frames(video_path, output_folder)
print(f"Extracted {len(frames)} frames and saved in {output_folder}")