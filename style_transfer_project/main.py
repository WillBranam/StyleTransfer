import os
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from models.transformer import TransformerNet
from utils.losses import compute_content_loss, compute_style_loss
from datasets.wikiart_csv.wikiart_dataset import WikiArtCSVLoader

# --- CONFIG ---
csv_file = "datasets/wikiart_csv/style_train.csv"
style_root = "images/wikiart_all"
content_dir = "images/content"
model_path = "transformer.pth"
image_size = 256
epochs = 5
batch_size = 4

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])

def extract_features(x, vgg):
    features = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in {3, 8, 17, 26}:
            features.append(x)
    return features

def train_model():
    print("Starting training...")
    dataset = WikiArtCSVLoader(csv_file, style_root, content_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerNet().to(device)
    vgg = models.vgg19(pretrained=True).features[:21].eval().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
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

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def stylize_image(content_path, style_path, output_path="stylized_output.jpg"):
    print("Stylizing image...")
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def load_image(img_path):
        image = Image.open(img_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)

    content = load_image(content_path)
    with torch.no_grad():
        output = model(content)
        output = torch.clamp(output, 0, 1)

    out_image = transforms.ToPILImage()(output.squeeze().cpu())
    out_image.save(output_path)
    print(f"Stylized image saved to {output_path}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    train_model()

    # Prompt user for input
    print("\n--- Stylize Your Own Image ---")
    content_path = input("Enter path to content image: ").strip()
    style_path = input("Enter path to style image (only used to prompt, not for inference): ").strip()
    output_path = input("Enter output file name (e.g., output.jpg): ").strip() or "stylized_output.jpg"

    if not os.path.isfile(content_path):
        print("Content image not found. Exiting.")
    else:
        stylize_image(content_path, style_path, output_path)
