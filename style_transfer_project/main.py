import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from models.transformer import TransformerNet
from utils.losses import compute_content_loss, compute_style_loss

# --- CONFIG ---
content_dir = "images/content"
style_path = "images/style_selected/starry.jpg"
model_path = "transformer_fixedstyle.pth"
image_size = 256
epochs = 5
batch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the fixed style image
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])
style_image = Image.open(style_path).convert("RGB")
fixed_style_tensor = transform(style_image).unsqueeze(0).to(device)

# Dataset using only content images
class ContentOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, content_dir):
        self.content_paths = [
            os.path.join(content_dir, f)
            for f in os.listdir(content_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content = Image.open(self.content_paths[idx]).convert("RGB")
        return self.transform(content)

def extract_features(x, vgg):
    feats = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in {3, 8, 17, 26}:
            feats.append(x)
    return feats

def train_model():
    from torchvision import models

    dataset = ContentOnlyDataset(content_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerNet().to(device)
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:21].eval().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for content in loader:
            content = content.to(device)
            style = fixed_style_tensor.expand(content.size(0), -1, -1, -1)
            with torch.no_grad():
                style_feats = extract_features(style, vgg)

            stylized = model(content)

            content_feats = extract_features(content, vgg)
            stylized_feats = extract_features(stylized, vgg)

            c_loss = compute_content_loss(stylized_feats, content_feats)
            s_loss = compute_style_loss(stylized_feats, style_feats)
            total_loss = c_loss + 10.0 * s_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {total_loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

def stylize_image(content_path, output_name="stylized_output.jpg"):
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(content).squeeze().clamp(0, 1).cpu()

    os.makedirs("images/output", exist_ok=True)
    output_path = os.path.join("images/output", output_name)
    transforms.ToPILImage()(output).save(output_path)
    print(f"Stylized image saved to {output_path}")

if __name__ == "__main__":
    train_model()
    content_path = input("Enter path to content image: ").strip()
    output_name = input("Enter output file name [stylized_output.jpg]: ").strip() or "stylized_output.jpg"
    if not os.path.isfile(content_path):
        print("Content image not found.")
    else:
        stylize_image(content_path, output_name)
