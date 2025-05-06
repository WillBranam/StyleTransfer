from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, content_dir, style_dir, image_size=256):
        self.content_paths = [os.path.join(content_dir, x) for x in os.listdir(content_dir)]
        self.style_paths = [os.path.join(style_dir, x) for x in os.listdir(style_dir)]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return min(len(self.content_paths), len(self.style_paths))

    def __getitem__(self, idx):
        content = self.transform(Image.open(self.content_paths[idx % len(self.content_paths)]).convert("RGB"))
        style = self.transform(Image.open(self.style_paths[idx % len(self.style_paths)]).convert("RGB"))
        return content, style
