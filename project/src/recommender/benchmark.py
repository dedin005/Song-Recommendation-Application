import os
import time
import torch
import argparse
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from models.model_resnet_smaller import Autoencoder as AutoencoderLowerRep
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Benchmark Autoencoder Model')
parser.add_argument('--model-path', type=str, default=None,
                    help='Path to saved model parameters')
args = parser.parse_args()

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])

batch_size = 46

# Paths to the data folders
data_folder = '../../data/mel_specs_resized_partitions'
test_path = os.path.join(data_folder, 'test')

# Create test dataloader
test_loader = DataLoader(datasets.ImageFolder(
    root=test_path, transform=transform), batch_size=batch_size, shuffle=False)

# Model Initialization
model = AutoencoderLowerRep(models.resnet34).to(device)

# Load model parameters if provided
if args.model_path:
    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model parameters from {args.model_path}")

criterion = nn.MSELoss()

# Benchmarking
model.eval()
test_loss = 0
total_inference_time = 0
total_ssim = 0
total_psnr = 0

with torch.no_grad():
    for i, (img, _) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Benchmarking"):
        start_time = time.time()
        output = model(img.to(device))
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        loss = criterion(output, img.to(device))
        test_loss += loss.item()

        output_np = output.cpu().numpy()[:, :3, :, :]  # Exclude alpha channel
        img_np = img.numpy()[:, :3, :, :]             # Exclude alpha channel

        output_np = output_np.transpose(0, 2, 3, 1)  # Convert to NHWC format
        img_np = img_np.transpose(0, 2, 3, 1)        # Convert to NHWC format

        for j in range(output_np.shape[0]):  # Loop over each image in the batch
            total_ssim += ssim(output_np[j], img_np[j], data_range=1.0, channel_axis=-1)
            total_psnr += psnr(output_np[j], img_np[j], data_range=1.0)

        # Save a few example images
        if i < 3:
            vutils.save_image(output.cpu(), f"../../benchmark_output_batch_{i+1}.png")

avg_test_loss = test_loss / len(test_loader)
avg_inference_time = total_inference_time / len(test_loader)
num_images = len(test_loader.dataset)
avg_ssim = total_ssim / num_images
avg_psnr = total_psnr / num_images

print(f"Test Loss: {avg_test_loss:.6f}")
print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
print(f"Average SSIM: {avg_ssim:.6f}")
print(f"Average PSNR: {avg_psnr:.6f}")

# Peak Memory Usage (for CUDA)
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")

# Model Size
model_size = os.path.getsize(args.model_path) / (1024 ** 2)  # Convert to MB
print(f"Model Size: {model_size:.2f} MB")
