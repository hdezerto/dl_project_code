import torch
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Custom Dataset wrapper to apply a specific transform ---
class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
class PseudoLabelledDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx].item()  # .item() ensures label is an int

    def __len__(self):
        return len(self.images)

def generate_pseudo_labels(model, unlabelled_loader):
    model.eval()
    pseudo_images = []
    pseudo_labels = []
    with torch.no_grad():
        for imgs, _ in unlabelled_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            pseudo_images.append(imgs.cpu())
            pseudo_labels.append(preds.cpu())
    pseudo_images = torch.cat(pseudo_images)
    pseudo_labels = torch.cat(pseudo_labels)
    return PseudoLabelledDataset(pseudo_images, pseudo_labels)

# --- Subsample dataset by fraction, stratified by class ---
def subsample_dataset(dataset, fraction):
    targets = np.array([y for _, y in dataset])
    classes = np.unique(targets)
    indices = []
    for c in classes:
        class_indices = np.where(targets == c)[0]
        n_samples = max(1, int(len(class_indices) * fraction))
        indices.extend(np.random.choice(class_indices, n_samples, replace=False))
    np.random.shuffle(indices)
    return Subset(dataset, indices)

# --- Training loop ---
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.4f}")

# --- Evaluation ---
def test_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

# --- Load Oxford-IIIT Pet Dataset for multi-class classification ---
base_train_val_dataset = OxfordIIITPet(root='./dataset', split='trainval', target_types='category', download=True)
test_dataset_raw = OxfordIIITPet(root='./dataset', split='test', target_types='category', download=True)

# --- Split train/val ---
num_train_val_samples = len(base_train_val_dataset)
val_fraction = 0.2
num_val_samples = int(val_fraction * num_train_val_samples)
num_train_samples_for_split = num_train_val_samples - num_val_samples
train_subset_raw, val_subset_raw = torch.utils.data.random_split(base_train_val_dataset, [num_train_samples_for_split, num_val_samples])

# --- Apply transforms ---
train_subset = TransformedDataset(train_subset_raw, transform=transform)
val_subset = TransformedDataset(val_subset_raw, transform=transform)
test_dataset = TransformedDataset(test_dataset_raw, transform=transform)

val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

num_classes = 37

# --- Fractions to test ---
fractions = [0.01, 0.1, 0.5, 1.0]
results = []

for frac in fractions:
    print(f"\n--- Fraction of labelled data: {frac*100:.1f}% ---")
    # 1. Subsample labelled data
    labelled_subset = subsample_dataset(train_subset, frac)
    labelled_loader = DataLoader(labelled_subset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    # 2. Unlabelled data (the rest)
    all_indices = set(range(len(train_subset)))
    labelled_indices = set(labelled_subset.indices)
    unlabelled_indices = list(all_indices - labelled_indices)
    unlabelled_subset = Subset(train_subset, unlabelled_indices)
    unlabelled_loader = DataLoader(unlabelled_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Train on labelled data
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_model(model, labelled_loader, optimizer, criterion, num_epochs=5)

    if len(unlabelled_subset) > 0:
        pseudo_labelled_dataset = generate_pseudo_labels(model, unlabelled_loader)
        # 5. Combine and retrain
        combined_dataset = ConcatDataset([labelled_subset, pseudo_labelled_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        train_model(model, combined_loader, optimizer, criterion, num_epochs=3)

    # 6. Evaluate
    acc = test_model(model, test_loader)
    results.append((frac, acc))
    print(f"Fraction: {frac}, Test Accuracy: {acc:.4f}%")

# --- Plot results ---
fractions, accuracies = zip(*results)
plt.plot([f*100 for f in fractions], accuracies, marker='o')
plt.xlabel('Percentage of Labelled Data Used')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Labelled Data Fraction (with Pseudo-Labeling)')
plt.grid(True)
plt.show()