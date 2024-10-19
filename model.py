import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from linformer import Linformer

BATCH_SIZE = 8

class MultiModalNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalNet, self).__init__()

        # Audio processing
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(51200, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fc_audio1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.fc_audio2 = nn.Sequential(
            nn.Linear(128, num_classes)
        )

        # self.audio_conv = models.resnet18()
        # self.audio_conv.fc = nn.Linear(self.audio_conv.fc.in_features, 128)


        # Video processing using ResNet
        self.video_resnet = models.resnet18(pretrained=True)
        self.video_resnet.fc = nn.Linear(self.video_resnet.fc.in_features, 128)

        # weighted importance
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        # Fully connected layers
        self.fc_combined = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.current_attn = torch.zeros((BATCH_SIZE, 2))

    def forward(self, audio, video, audio_only):
        # Audio branch
        audio_features = self.audio_conv(audio)
        audio_features = self.fc_audio1(audio_features)

        if not audio_only:
            # Video branch
            video_features = self.video_resnet(video)

            # Concatenate features
            combined_features = torch.cat((audio_features, video_features), dim=1)

            # Apply attention
            attention_weights = self.attention(combined_features)
            self.current_attn = attention_weights

            # print("==Forward start==")
            # print(attention_weights)
            # print("====")

            first_part = attention_weights[:, 0].unsqueeze(1).expand(-1, 128)
            second_part = attention_weights[:, 1].unsqueeze(1).expand(-1, 128)
            # Concatenate the two parts
            attention_weights = torch.cat((first_part, second_part), dim=1)

            weighted_features = combined_features * attention_weights

            # Fully connected layers
            output = self.fc_combined(weighted_features)
        else:
            output = self.fc_audio2(audio_features)

        return output
    
    def attention_regularization(self):
        # L2 penalty on the difference between the attention weights
        reg_loss = torch.sum((self.current_attn[:, 0] - self.current_attn[:, 1]) ** 2)
        return reg_loss
