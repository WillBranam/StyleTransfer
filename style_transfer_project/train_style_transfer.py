import torch
from torch.utils.data import DataLoader
from models.transformer import TransformerNet
from utils.losses import compute_content_loss, compute_style_loss
from datasets.wikiart_dataset import ImageDataset
import torchvision.models as models

dataset = ImageDataset("images/content", "images/wikiart")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNet().to(device)
vgg = models.vgg19(pretrained=True).features[:21].eval().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for content, style in loader:
        content, style = content.to(device), style.to(device)
        stylized = model(content)

        def extract_features(x): return [vgg[:i+1](x) for i in [3, 8, 17, 26]]
        content_features = extract_features(content)
        style_features = extract_features(style)
        stylized_features = extract_features(stylized)

        c_loss = compute_content_loss(stylized_features, content_features)
        s_loss = compute_style_loss(stylized_features, style_features)
        total_loss = c_loss + 10.0 * s_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
