import os
from PIL import Image
from torch.utils.data import Dataset

class MusicImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.labels = sorted(os.listdir(root_dir))
        self.files = []
        for label in self.labels:
            label_dir = os.path.join(root_dir, label)
            for file in os.listdir(label_dir):
                self.files.append((os.path.join(label_dir, file), label))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels.index(label)

