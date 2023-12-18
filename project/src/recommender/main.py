import os
import time
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.model_resnet_smaller import Autoencoder as AutoencoderLowerRep

# Argument parser
parser = argparse.ArgumentParser(description='Train Autoencoder Model')
parser.add_argument('--model-path', type=str, default=None,
                    help='Path to saved model parameters')
args = parser.parse_args()

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])

# Hyperparameters
batch_size = 47
epochs = 500
model_path = '../../models/recommender'

# Paths to the data folders
data_folder = '../../data/mel_specs_resized_partitions'
train_path, val_path, test_path = [os.path.join(data_folder, x) for x in [
    'validation', 'test', 'train']]

# Create datasets and dataloaders
train_loader = DataLoader(datasets.ImageFolder(
    root=train_path, transform=transform), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(datasets.ImageFolder(
    root=val_path, transform=transform), batch_size=batch_size, shuffle=False)
# single_test_loader = DataLoader(datasets.ImageFolder(
#     root='../../data/tests/', transform=transform), batch_size=1, shuffle=False)

# Initialize the model, loss, and optimizer
model_mapping = {
    'resnet_smaller': {'model': models.resnet34, 'autoencoder': AutoencoderLowerRep},
}

# Function to get model and autoencoder
def get_model_and_autoencoder(model_name):
    if model_name in model_mapping:
        resnet = model_mapping[model_name]['model']
        Autoencoder = model_mapping[model_name]['autoencoder']

        autoencoder = Autoencoder(resnet).to(device)

        return autoencoder, resnet
    else:
        raise ValueError("Invalid model name")


model_name = 'resnet_smaller'
model, resnet = get_model_and_autoencoder(model_name)


# Load model parameters if provided
if args.model_path:
    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model parameters from {args.model_path}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Initialize the best validation loss to a high value
best_val_loss = float('inf')  # 0.004702

# Training loop
for epoch in range(epochs):
    model.train()
    start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0

    for i, data in enumerate(train_loader, 0):
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
        if (i+1) % 100 == 0:
            print(f"   Train Batch {i+1:4d}, Loss: {loss.item():.6f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            img, _ = data
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            val_loss += loss.item()
            val_batches += 1

    # Calculate average losses
    avg_train_loss = epoch_loss / num_batches
    avg_val_loss = val_loss / val_batches

    # Check for best validation loss
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     torch.save(model.state_dict(),
    #                f"{model_path}/{resnet.__name__}_deconv_best_{best_val_loss:.4f}.pth")
    #     print(
    #         f"   New best model saved with validation loss: {best_val_loss:.6f}")

    # with torch.no_grad():
    #     for i, (img, _) in enumerate(single_test_loader):
    #         img = img.to(device)
    #         output = model(img)
    #         output = output.cpu().squeeze(0)  # Remove the batch dimension
    #         vutils.save_image(output, f"../../output_{epoch+1}.png")

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch [{epoch+1:4d}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s")
