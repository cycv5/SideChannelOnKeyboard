import cv2
import librosa
import soundfile as sf
import os
from preprocess import homography_and_crop
import numpy as np

def extract_audio(audio_path, start_time, end_time, output_audio_path):
    y, sr = librosa.load(audio_path)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    key_press_audio = y[start_sample:end_sample]

    sf.write(output_audio_path, key_press_audio, sr)


def extract_frame(video_path, time_point, output_image_path):

    # Get the frame at the specified time point
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
    ret, frame = cap.read()
    cap.release()

    # Convert the frame (numpy array) to an image and save it
    src_points = np.array([(1141, 250), (474, 510), (354, 362), (960, 186)],
                          dtype=np.int32)  # Example coordinates of keyboard corners
    dst_points = np.array([[0, 0], [515, 0], [515, 200], [0, 200]], dtype=np.int32)  # Desired rectangle coordinates
    cropped_frame, h_matrix = homography_and_crop(frame, src_points, dst_points)
    cv2.imwrite(output_image_path, cropped_frame)


# start_time = 5.37  # in seconds
# end_time = 5.66  # in seconds

start_time = 6.24  # in seconds
end_time = 6.53  # in seconds

time_point = (start_time + end_time) / 2.0 - 0.01  # in seconds

# Example usage:
video_path = "raw/D.mp4"
audio_path = "raw/D.wav"
output_audio_path = f"D_pressed_{start_time}_{end_time}.wav"
output_image_path = f"D_pressed_{start_time}_{end_time}_img.png"


extract_audio(audio_path, start_time, end_time, output_audio_path)
extract_frame(video_path, time_point, output_image_path)

print(f"Audio saved to {output_audio_path}")
print(f"Frame saved to {output_image_path}")






# Function to capture mouse click events
# def get_coordinates(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         coordinates.append((x, y))
#         print(f"Coordinate captured: ({x}, {y})")
#
# # Initialize list to store coordinates
# coordinates = []
#
# # Load the video
# video_path = "raw/C.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Read the first frame
# ret, frame = cap.read()
# cap.release()
#
# # Display the frame and set the mouse callback function
# cv2.imshow("Frame", frame)
# cv2.setMouseCallback("Frame", get_coordinates)
#
# print("Click on the four corners of the keyboard in the video frame.")
# print("Press 'q' to quit after selecting the points.")
#
# while True:
#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
#
# # Print the captured coordinates
# print("Captured coordinates:", coordinates)