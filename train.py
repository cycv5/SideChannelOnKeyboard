import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
from model import MultiModalNet
import matplotlib.pyplot as plt
from dataloader import KeystrokeDataset

train_losses = []
val_losses = []
train_acc = []
val_acc = []

device = "cuda:0"

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor()
])

# parameters
BATCH_SIZE = 8

num_epochs = int(input("Enter the number of epochs: "))
use_existing_weight = True
if input("Use NEW weight (saved weight could be ERASED, pls back up)? (y/n): ") == 'y':
    use_existing_weight = False
audio_only = False
if input("Train audio only? (y/n): ") == 'y':
    audio_only = True

INPUT_WEIGHTS = 'audio_weights.pth'
OUTPUT_WEIGHTS = 'combined_weights.pth'
if audio_only:
    OUTPUT_WEIGHTS = 'audio_weights.pth'

LR = 0.0001
if audio_only:
    LR = 0.001


# Data loaders
train_dataset = KeystrokeDataset(audio_dir='train/audio', video_dir='train/video', transform=transform)
valid_dataset = KeystrokeDataset(audio_dir='valid/audio', video_dir='valid/video', transform=transform)
# test_dataset = KeystrokeDataset(audio_dir='test/audio', video_dir='test/video', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss function, and optimizer
num_classes = 27
model = MultiModalNet(num_classes).to(device)
if use_existing_weight:
    model.load_state_dict(torch.load(INPUT_WEIGHTS))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=25):
    print(f"Using GPU: {torch.cuda.is_available()}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_count = 0
        count = 0
        for audio, video, labels in train_loader:
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio, video, audio_only)
            loss = criterion(outputs, labels)

            lambda_reg = 0 if audio_only else 0.04
            attention_reg_loss = model.attention_regularization()
            total_loss = loss + lambda_reg * attention_reg_loss

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += loss.item()
            output_class = torch.argmax(outputs, dim=1)
            compare = (output_class == labels)
            correct_count += compare.sum().item()
            count += labels.shape[0]

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_count = 0
        val_count = 0

        with torch.no_grad():
            for audio, video, labels in valid_loader:
                audio, video, labels = audio.to(device), video.to(device), labels.to(device)
                outputs = model(audio, video, audio_only)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                output_class = torch.argmax(outputs, dim=1)
                compare = (output_class == labels)
                val_correct_count += compare.sum().item()
                val_count += labels.shape[0]
                if val_count < 9:
                    print(f"Example outputs and label (validation): \n{output_class}\n{labels}")


        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss Bavg: {running_loss/len(train_loader)}, Valid Loss Bavg: {val_loss/len(valid_loader)}')
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss/len(valid_loader))
        print(
            f'Train acc: {100*(correct_count/count):.4f}%, Valid acc: {100*(val_correct_count/val_count):.4f}% \n')
        train_acc.append(correct_count/count)
        val_acc.append(val_correct_count/val_count)

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)

# Save the model
torch.save(model.state_dict(), OUTPUT_WEIGHTS)

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Acc')
plt.plot(val_acc, label='Validation Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()