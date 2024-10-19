import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import MultiModalNet
import matplotlib.pyplot as plt
from dataloader import KeystrokeDataset
from collections import defaultdict
import random
import openai


device = "cuda:0"
print("Please add your API key")
openai.api_key = ""
# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# parameters
BATCH_SIZE = 8

INPUT_WEIGHTS = 'combined_weights.pth'


# Data loaders
test_dataset = KeystrokeDataset(audio_dir='test/audio', video_dir='test/video', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_semantic_dataset = KeystrokeDataset(audio_dir='test/audio', video_dir='test/video', transform=transform)
test_semantic_by_letter = defaultdict(list)
for audio, video, label in test_semantic_dataset:
    letter = chr(ord('A') + label) if label < 26 else " "
    test_semantic_by_letter[letter].append((audio, video, label))


num_classes = 27
model = MultiModalNet(num_classes).to(device)

model.load_state_dict(torch.load(INPUT_WEIGHTS))

criterion = nn.CrossEntropyLoss()


# Pre-trained GPT for sentence correction
def getGPTResponse(recovered_msg):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You do what you are told."},
            {"role": "user", "content": "Please check for errors and respond with a correct and clean version of the following (and only the following): " + recovered_msg}
        ]
    )

    return response.choices.message['content']


# test function
def test_model(model, test_loader, criterion):
    print(f"Using GPU: {torch.cuda.is_available()}")
    print("Testing model on random letters.")
    # Validation
    model.eval()
    test_loss = 0.0
    test_correct_count = 0
    test_count = 0

    with torch.no_grad():
        for audio, video, labels in test_loader:
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            outputs = model(audio, video, audio_only=False)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            output_class = torch.argmax(outputs, dim=1)
            compare = (output_class == labels)
            test_correct_count += compare.sum().item()
            test_count += labels.shape[0]
    print(f"Test accuracy: {100*(test_correct_count/test_count):.4f}% \n")


# Test the model
test_model(model, test_loader, criterion)


# Test semantic
def test_semantic(model):
    print(f"Using GPU: {torch.cuda.is_available()}")
    print("Testing model on sentences.")

    target = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    recovered = ""
    print(f"Target to recover: {target}")
    # Validation
    model.eval()
    test_correct_count = 0
    test_count = 0

    with torch.no_grad():
        for c in target:
            audio, video, label = test_semantic_by_letter[c][random.randint(0, len(test_semantic_by_letter[c]) - 1)]
            audio, video = audio.unsqueeze(0).to(device), video.unsqueeze(0).to(device)
            outputs = model(audio, video, audio_only=False)
            output_class = torch.argmax(outputs, dim=1).item()
            test_correct_count += 1 if output_class == label else 0
            test_count += 1
            recovered += chr(ord("A") + output_class) if output_class < 26 else " "

    print(f"Sentence test accuracy: {100*(test_correct_count/test_count):.4f}% \n")
    print(f"Recovered sentence is {recovered}")
    if openai.api_key != "":
        gpt_response = getGPTResponse(recovered)
        print(f"Recovered sentence through pre-trained GPT: {gpt_response}")


test_semantic(model)

# plt.figure(figsize=(10, 5))
# plt.plot(train_acc, label='Training Acc')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Training and Validation Accuracy')
# plt.show()