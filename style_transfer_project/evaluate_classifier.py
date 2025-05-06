import os
import csv
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

# Dataset from CSV
class WikiArtStyleClassificationDataset(Dataset):
    def __init__(self, csv_file, image_root, class_file):
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            self.samples = [row for row in reader if row]

        with open(class_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label_idx = self.samples[idx]
        image_path = os.path.join(self.image_root, rel_path)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(label_idx)

# Paths
csv_path = "datasets/wikiart_csv/style_train.csv"
image_root = "images/wikiart_all"
class_txt = "datasets/wikiart_csv/style_class.txt"

# Dataset and DataLoader
dataset = WikiArtStyleClassificationDataset(csv_path, image_root, class_txt)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
classifier = models.resnet18(pretrained=True)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(dataset.classes))
classifier = classifier.to(device)

# Optimizer and Loss
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    classifier.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = classifier(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# Evaluation
classifier.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = classifier(imgs)
        _, preds = torch.max(out, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=dataset.classes))
