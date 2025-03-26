import os
import random
import shutil

DATASET_PATH = r"E:\DataSets\tarangDataset\bharatnatyam\extracted"

# Adjust ratios as needed
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def split_data():
    train_path = os.path.join(DATASET_PATH, "Training")
    val_path = os.path.join(DATASET_PATH, "Valid")
    test_path = os.path.join(DATASET_PATH, "Test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Loop through each Adavu
    for adavu in os.listdir(DATASET_PATH):
        adavu_path = os.path.join(DATASET_PATH, adavu)
        if adavu in ["Training", "Test", "Valid"] or not os.path.isdir(adavu_path):
            continue

        # Create subfolders for this Adavu in Training/Valid/Test
        adavu_train = os.path.join(train_path, adavu)
        adavu_val = os.path.join(val_path, adavu)
        adavu_test = os.path.join(test_path, adavu)
        os.makedirs(adavu_train, exist_ok=True)
        os.makedirs(adavu_val, exist_ok=True)
        os.makedirs(adavu_test, exist_ok=True)

        # Inside each Adavu, we have multiple video subfolders
        for video_subfolder in os.listdir(adavu_path):
            video_path = os.path.join(adavu_path, video_subfolder)
            if not os.path.isdir(video_path):
                continue

            frames = [f for f in os.listdir(video_path) if f.lower().endswith(".jpg")]
            random.shuffle(frames)

            total_frames = len(frames)
            train_count = int(total_frames * TRAIN_RATIO)
            val_count = int(total_frames * VAL_RATIO)
            test_count = total_frames - train_count - val_count

            train_frames = frames[:train_count]
            val_frames = frames[train_count:train_count + val_count]
            test_frames = frames[train_count + val_count:]

            # Move frames
            for f in train_frames:
                src = os.path.join(video_path, f)
                dst = os.path.join(adavu_train, f"{video_subfolder}_{f}")
                shutil.move(src, dst)

            for f in val_frames:
                src = os.path.join(video_path, f)
                dst = os.path.join(adavu_val, f"{video_subfolder}_{f}")
                shutil.move(src, dst)

            for f in test_frames:
                src = os.path.join(video_path, f)
                dst = os.path.join(adavu_test, f"{video_subfolder}_{f}")
                shutil.move(src, dst)

            print(f"{adavu}/{video_subfolder}: {train_count} train, {val_count} valid, {test_count} test")

def main():
    split_data()

if __name__ == "__main__":
    main()
    print("✅ Splitting complete!")
