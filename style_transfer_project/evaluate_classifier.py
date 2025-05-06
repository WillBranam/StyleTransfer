import torch
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

dataset = ImageFolder("images/wikiart", transform=transform)
loader = DataLoader(dataset, batch_size=32)

classifier = models.resnet18(pretrained=True)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(dataset.classes))
classifier = classifier.to(device)

# Dummy training loop (to be replaced with actual training if needed)
# This only demonstrates structure
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = classifier(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# After training, evaluate
classifier.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = classifier(imgs)
        _, preds = torch.max(out, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=dataset.classes))
