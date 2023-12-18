import torch
import torchvision.models as models
from models.model_resnet_smaller import Autoencoder
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = Autoencoder(models.resnet34).to(device)
model_path = '../../models/recommender/resnet34_deconv_best_0.0029.pth'

# Load the full model
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Extract encoder and downsample layers
encoder_state_dict = {k: v for k, v in model.state_dict().items() if 'encoder' in k or 'downsample' in k}

# Save the encoder and downsample layers
encoder_path = '../../models/recommender/encoder.pth'
os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
torch.save(encoder_state_dict, encoder_path)
