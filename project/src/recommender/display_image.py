import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.transforms import functional as F
from models.model import Autoencoder
import torchvision.models as models

import os

class CustomCrop:
    def __call__(self, img):
        return F.crop(img, 16, 0, 384, 992)  # top, left, height, width

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize the model and load the pretrained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(models.resnet34).to(device)
model.load_state_dict(torch.load('../../models/recommender/resnet34_deconv_best_0.0029.pth'))
model.eval()

# Load the test images
test_dataset = datasets.ImageFolder(root=os.path.join('../../tests/'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run model on test data and save the output images
with torch.no_grad():
    for i, (img, _) in enumerate(test_loader):
        img = img.to(device)
        output = model(img)
        
        output = output.cpu().squeeze(0)  # Remove the batch dimension
        vutils.save_image(output, f"../../test{i+1}.png")

print("Test images saved.")
