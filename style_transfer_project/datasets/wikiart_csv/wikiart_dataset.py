import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class WikiArtCSVLoader(Dataset):
    def __init__(self, csv_file, image_root, content_dir, image_size=256):
        self.image_root = image_root
        self.content_dir = content_dir
        self.image_paths = []

        # Read image paths from the CSV
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            self.image_paths = [row[0] for row in reader if row]

        # Collect all content image paths
        self.content_paths = [
            os.path.join(content_dir, fname)
            for fname in os.listdir(content_dir)
            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return min(len(self.image_paths), len(self.content_paths))

    def __getitem__(self, idx):
        style_path = os.path.join(self.image_root, self.image_paths[idx])
        content_path = random.choice(self.content_paths)

        style_image = self.transform(Image.open(style_path).convert("RGB"))
        content_image = self.transform(Image.open(content_path).convert("RGB"))

        return content_image, style_image
