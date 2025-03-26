#Will move the frames from the extracted folder to Training Valid and Test folders
import os
import random
import shutil

DATASET_PATH = r"E:\DataSets\tarangDataset\odissi\extracted"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def split_data():
    train_dir = os.path.join(DATASET_PATH, "Training")
    valid_dir = os.path.join(DATASET_PATH, "Valid")
    test_dir = os.path.join(DATASET_PATH, "Test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category_name in os.listdir(DATASET_PATH):
        cat_path = os.path.join(DATASET_PATH, category_name)
        if category_name in ["Training", "Valid", "Test"] or not os.path.isdir(cat_path):
            continue

        cat_train = os.path.join(train_dir, category_name)
        cat_valid = os.path.join(valid_dir, category_name)
        cat_test = os.path.join(test_dir, category_name)

        os.makedirs(cat_train, exist_ok=True)
        os.makedirs(cat_valid, exist_ok=True)
        os.makedirs(cat_test, exist_ok=True)

        for video_sub in os.listdir(cat_path):
            video_sub_path = os.path.join(cat_path, video_sub)
            if not os.path.isdir(video_sub_path):
                continue

            frames = [f for f in os.listdir(video_sub_path) if f.lower().endswith('.jpg')]
            random.shuffle(frames)

            total = len(frames)
            train_count = int(total * TRAIN_RATIO)
            val_count = int(total * VAL_RATIO)
            test_count = total - train_count - val_count

            train_frames = frames[:train_count]
            val_frames = frames[train_count:train_count + val_count]
            test_frames = frames[train_count + val_count:]

            # Move frames
            for f in train_frames:
                src = os.path.join(video_sub_path, f)
                dst = os.path.join(cat_train, f"{video_sub}_{f}")
                shutil.move(src, dst)

            for f in val_frames:
                src = os.path.join(video_sub_path, f)
                dst = os.path.join(cat_valid, f"{video_sub}_{f}")
                shutil.move(src, dst)

            for f in test_frames:
                src = os.path.join(video_sub_path, f)
                dst = os.path.join(cat_test, f"{video_sub}_{f}")
                shutil.move(src, dst)

            print(f"{category_name}/{video_sub}: {train_count} train, {val_count} val, {test_count} test")

def main():
    split_data()

if __name__ == "__main__":
    main()
    print("Splitting Done")
