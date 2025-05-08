import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random

from models.transformer import TransformerNet
from utils.losses import compute_content_loss, compute_style_loss

# Fallback style image generator using random noise (placeholder for GAN)
def generate_style_image(output_path, image_size=256):
    from torchvision.utils import save_image
    tensor = torch.rand(3, image_size, image_size)
    save_image(tensor, output_path)

class SimpleStyleDataset(torch.utils.data.Dataset):
    def __init__(self, content_dir, style_dir, image_size=256):
        self.content_paths = [
            os.path.join(content_dir, f)
            for f in os.listdir(content_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.style_paths = [
            os.path.join(style_dir, f)
            for f in os.listdir(style_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return min(len(self.content_paths), len(self.style_paths))

    def __getitem__(self, idx):
        content = Image.open(self.content_paths[idx]).convert("RGB")
        style = Image.open(random.choice(self.style_paths)).convert("RGB")
        return self.transform(content), self.transform(style)

def extract_features(x, vgg):
    feats = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in {3, 8, 17, 26}:
            feats.append(x)
    return feats

def train_model():
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_dir = "images/content"
    style_dir = "images/style_generated"
    os.makedirs(style_dir, exist_ok=True)

    if not os.listdir(style_dir):
        print("No style images found. Generating from StyleGAN2...")
        import subprocess
        subprocess.run(["python3", "images/generate_style_images_tf.py"])


    dataset = SimpleStyleDataset(content_dir, style_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TransformerNet().to(device)
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:21].eval().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        for content, style in loader:
            content, style = content.to(device), style.to(device)
            stylized = model(content)

            content_feats = extract_features(content, vgg)
            style_feats = extract_features(style, vgg)
            stylized_feats = extract_features(stylized, vgg)

            c_loss = compute_content_loss(stylized_feats, content_feats)
            s_loss = compute_style_loss(stylized_feats, style_feats)
            total_loss = c_loss + 10.0 * s_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {total_loss.item():.4f}")
    torch.save(model.state_dict(), "transformer.pth")

def stylize_image(content_path, output_path="stylized_output.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load("transformer.pth", map_location=device))
    model.eval()

    content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(content).squeeze().clamp(0, 1).cpu()
    transforms.ToPILImage()(output).save(output_path)
    print(f"Stylized image saved to {output_path}")

if __name__ == "__main__":
    train_model()
    content_path = input("Enter path to content image: ").strip()
    output_path = input("Enter output file name [stylized_output.jpg]: ").strip() or "stylized_output.jpg"
    if not os.path.isfile(content_path):
        print("Error: Content image not found.")
    else:
        stylize_image(content_path, output_path)
