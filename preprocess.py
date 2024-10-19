import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import moviepy.editor as mp_edit
import mediapipe as mp
import cv2
import torchaudio
import random
import torch

def separate_key_presses(audio_path, output_dir, input_letter, threshold, max_threshold, min_duration=0.15):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    # Compute the short-time Fourier transform (STFT)
    ft = np.abs(librosa.stft(y))

    # Compute the root-mean-square (RMS) energy for each frame
    rms = librosa.feature.rms(S=ft)[0]
    rms_times = librosa.frames_to_time([0, 1], sr=sr)
    print(rms_times)
    frame_time = rms_times[1] - rms_times[0]
    # Find the frames where the RMS energy exceeds the threshold
    key_press_frames = np.where(rms > threshold)[0]

    # Convert frames to time
    key_press_times = librosa.frames_to_time(key_press_frames, sr=sr)
    print(f"frame_time: {frame_time}")

    # Separate individual key presses
    key_press_intervals = []
    start_time, start_frame = key_press_times[0], key_press_frames[0]
    cur_interval_max = rms[start_frame]
    for i in range(1, len(key_press_times)):
        cur_interval_max = max(cur_interval_max, rms[key_press_frames[i-1]])
        if key_press_times[i] - start_time < min_duration:
            continue
        if key_press_times[i] - key_press_times[i - 1] > 0.1:
            end_time = key_press_times[i - 1]
            end_frame = key_press_frames[i - 1]
            start_time_compensator, end_time_compensator = 0.01, 0.01
            if start_frame > 0 and rms[start_frame-1] < rms[start_frame]:
                start_time_compensator += frame_time
            if end_frame < (len(rms) - 1) and rms[end_frame+1] < rms[end_frame]:
                end_time_compensator += frame_time
            if end_frame < (len(rms) - 2) and rms[end_frame+2] < rms[end_frame+1]:
                end_time_compensator += frame_time
            if cur_interval_max >= max_threshold:
                key_press_intervals.append((start_time-start_time_compensator, end_time+end_time_compensator))
                print(f"start_frame: {start_frame}, end_frame: {end_frame}")
                print(f"start_time: {start_time} + start_time_compensator: {start_time_compensator}, end_time: {end_time} + end_time_compensator: {end_time_compensator}")
            start_time = key_press_times[i]
            start_frame = key_press_frames[i]
            cur_interval_max = rms[start_frame]
    if cur_interval_max >= max_threshold:
        key_press_intervals.append((start_time, key_press_times[-1]))

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save individual key press audio files with start and end times in the file name
    for i, (start, end) in enumerate(key_press_intervals):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        key_press_audio = y[start_sample:end_sample]
        if len(input_letter) > 2:
            output_path = os.path.join(output_dir, f'{chr(ord("A") + i)}_pressed_{start:.2f}_{end:.2f}.wav')
        else:
            output_path = os.path.join(output_dir, f'{input_letter}_pressed_{start:.2f}_{end:.2f}.wav')
        sf.write(output_path, key_press_audio, sr)


def plot_rms_energy(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    # y = y[sr//2:-(sr//2)]  # cut non-keypress sounds

    # Compute the short-time Fourier transform (STFT)
    ft = np.abs(librosa.stft(y))

    # Compute the root-mean-square (RMS) energy for each frame
    rms = librosa.feature.rms(S=ft)[0]

    # Convert frames to time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # Plot the RMS energy
    plt.figure(figsize=(10, 6))
    plt.plot(times, rms, label='RMS Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Energy')
    plt.title('RMS Energy of Audio Signal')
    plt.legend()
    plt.show()


# Function to separate video and audio
def separate_video_audio(video_path, audio_path):
    video = mp_edit.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


# Function to detect hands and get fingertip positions
def get_finger_tips(video_path):
    cap = cv2.VideoCapture(video_path)
    finger_tips_positions = []
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_count += 1
                finger_tips = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                               hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                               hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                               hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                               hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
                finger_tips_positions.append(finger_tips)

        if hand_count == 1:
            finger_tips_positions.append([])

    cap.release()
    return finger_tips_positions


# Function to apply homography and crop the keyboard
def homography_and_crop(frame, src_points, dst_points):
    h, status = cv2.findHomography(src_points, dst_points)
    height, width = frame.shape[:2]
    warped_frame = cv2.warpPerspective(frame, h, (width, height))
    cropped_frame = warped_frame[dst_points[0][1]:dst_points[2][1], dst_points[0][0]:dst_points[1][0]]
    return cropped_frame, h


# Main function to process the video and audio files
def process_video_and_audio(video_path, audio_folder, letter, src_points, dst_points):
    samples = 0
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith(".wav"): # and audio_file.startswith(letter + "_"):
            samples += 1
            parts = audio_file.split("_")
            keypress_time = ((float(parts[2]) + float(parts[3][:-4])) / 2) - 0.01

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, keypress_time * 1000)
            ret, frame = cap.read()
            cap.release()

            cropped_frame, h_matrix = homography_and_crop(frame, src_points, dst_points)

            output_path = os.path.join(audio_folder, audio_file.replace(".wav", "_img.png"))
            cv2.imwrite(output_path, cropped_frame)


def audio_to_mel_spectrogram(audio_path, target_length):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Calculate the padding needed
    current_length = waveform.size(1)
    if current_length < target_length:
        padding = target_length - current_length
        left_padding = random.randint(0, padding)
        right_padding = padding - left_padding
        waveform = torch.nn.functional.pad(waveform, (left_padding, right_padding))
    elif current_length == target_length:
        waveform = waveform[:, :target_length]
    else:
        raise ValueError(f"Target length is smaller than the audio file. Length: {current_length}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
    return mel_spectrogram

# Example usage
if __name__ == '__main__':
    letter = "D"

    input_video_path = f"raw/{letter}.mp4"
    output_audio_path = f"raw/{letter}.wav"
    # output_video_path = f"raw/{letter}_hand.mp4"
    output_seg_audio_path = "dataset/"

    src_points = np.array([(1141, 250), (474, 510), (354, 362), (960, 186)],
                          dtype=np.int32)  # Example coordinates of keyboard corners
    dst_points = np.array([[0, 0], [515, 0], [515, 200], [0, 200]], dtype=np.int32)  # Desired rectangle coordinates

    # Separate video and audio
    # separate_video_audio(input_video_path, output_audio_path)
    print("Audio and video seperated!")

    # Separate key press sounds
    separate_key_presses(output_audio_path, output_seg_audio_path, letter, threshold=0.002, max_threshold=0.005)
    print("Audio key presses segmented!")

    # Process video with Mediapipe hand detection
    process_video_and_audio(input_video_path, output_seg_audio_path, letter, src_points, dst_points)
    print("Process Finished")

    plot_rms_energy(output_audio_path)

