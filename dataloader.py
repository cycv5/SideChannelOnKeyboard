import os
from torch.utils.data import DataLoader, Dataset
from preprocess import audio_to_mel_spectrogram
from PIL import Image
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class KeystrokeDataset(Dataset):
    def __init__(self, audio_dir, video_dir, transform=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.transform = transform
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = audio_file.split('_')[0]
        if label == "WS":
            intLabel = 26
        else:
            intLabel = ord(label) - ord('A')

        # Load audio
        audio_path = os.path.join(self.audio_dir, audio_file)
        mel_spectrogram = audio_to_mel_spectrogram(audio_path, target_length=20000)

        # Load video frame
        video_file = audio_file.replace('.wav', '_img.png')
        video_path = os.path.join(self.video_dir, video_file)
        image = Image.open(video_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if intLabel == 99999:  # do not show spectrogram unless asked. Valid: [0-26]
            _, sr = librosa.load(audio_path)
            a_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(np.squeeze(a_dB, 0), sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-Spectrogram')
            plt.tight_layout()
            plt.show()

        return mel_spectrogram, image, intLabel