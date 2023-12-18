from dataset import MusicImagesDataset
from model import Classifier

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import os

model_save_path = '../../models'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = MusicImagesDataset(
    root_dir='../../data/songs/images_cropped_segments', transform=transform)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size])

batch_size = 512//2
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model_type = models.resnet18  # resnet34
model_output_size = 512

criterion = nn.CrossEntropyLoss()
num_epochs = 1000

print(f"Training {model_type.__name__}")

model = Classifier(model_type, model_output_size, num_classes=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Adding the scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=50, factor=0.1, verbose=True)

best_val_loss = float('inf')

true_labels = []
pred_labels = []

def save_confusion_matrix(cm, epoch, path):
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion Matrix at Epoch {epoch+1}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(path, f'confusion_matrix_epoch.png'))
    plt.close()


for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    val_loss = 0
    true_labels.clear()
    pred_labels.clear()
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            
    val_loss /= len(val_dataloader)
    
    cm = confusion_matrix(true_labels, pred_labels)
    save_confusion_matrix(cm, epoch, model_save_path)

    # Update scheduler based on validation loss
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if val_loss < 0.45 and epoch > 50:
            torch.save(model.state_dict(), os.path.join(
                model_save_path, f'{model_type.__name__}_best_model.pth'))
            print(
                f"Model {model_type.__name__} saved at epoch {epoch+1} with validation loss of {val_loss:.6f}")
            # break
