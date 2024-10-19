import os
import shutil
import random

random.seed(2231)

# Define the paths
dataset_path = 'dataset'
train_path = 'train'
valid_path = 'valid'
test_path = 'test'

# Create directories if they don't exist
for path in [train_path, valid_path, test_path]:
    os.makedirs(os.path.join(path, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(path, 'video'), exist_ok=True)

# Get all unique keys (letters and WS)
keys = set(f.split('_')[0] for f in os.listdir(dataset_path) if f.endswith('.wav'))


# Function to copy files
def copy_files(files, dest_path):
    index = 0
    for file in files:
        base_name = os.path.splitext(file)[0]
        audio_file = f"{base_name}.wav"
        video_file = f"{base_name}_img.png"

        letter = base_name.split('_')[0]
        new_audio_file = f"{letter}_{index}.wav"
        new_video_file = f"{letter}_{index}_img.png"
        index += 1
        print(base_name)
        shutil.copy(os.path.join(dataset_path, audio_file), os.path.join(dest_path, 'audio', new_audio_file))
        shutil.copy(os.path.join(dataset_path, video_file), os.path.join(dest_path, 'video', new_video_file))


# Process each key
for key in keys:
    # Get all files for the current key
    key_files = [f for f in os.listdir(dataset_path) if f.startswith(key+"_") and f.endswith('.wav')]

    # Randomly shuffle the files
    random.shuffle(key_files)

    # Split into train, valid, and test sets
    train_files = key_files[:24]
    valid_files = key_files[24:27]
    test_files = key_files[27:30]

    # Copy the files to the respective directories
    copy_files(train_files, train_path)
    copy_files(valid_files, valid_path)
    copy_files(test_files, test_path)

print("Dataset organized successfully!")
