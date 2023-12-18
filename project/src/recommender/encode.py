import torch
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from models.model_resnet_smaller import Autoencoder
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to the data folders
data_folder = '../../data/mel_specs_resized_partitions'
data_base = '../../data'
# 'parts6to21', 'parts1to5', 'train', 'validation', 'test']
folders = ['parts31to46']

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Model initialization
model = Autoencoder(models.resnet34).to(device)
model_path = '../../models/recommender/resnet34_deconv_best_0.0029.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Process images and save embeddings
batch_size = 300  # Adjust as per your computational capability
num_workers = 1       # Adjust based on the number of CPU cores and available memory

for folder in folders:
    folder_path = os.path.join(data_folder, folder)
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)

    # Adjust DataLoader with num_workers and pin_memory
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    embeddings_dir = os.path.join(data_base, f'embeddings/{folder}')
    os.makedirs(embeddings_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(dataloader, desc=f"Processing {folder}")):
            images = images.to(device)
            encoded_imgs = model.encoder(images)
            downsampled_imgs = model.downsample(encoded_imgs)

            # Process and save embeddings for Milvus
            for j, embedding in enumerate(downsampled_imgs):
                # Flatten and convert to NumPy array
                flattened_embedding = embedding.view(-1).cpu().numpy()

                img_name = dataset.samples[i *
                                           batch_size + j][0].split('/')[-1]
                embedding_name = img_name.replace('.png', '_embedding.npy')

                # Save the NumPy array
                np.save(os.path.join(embeddings_dir, embedding_name),
                        flattened_embedding)
