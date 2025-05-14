#IMPROVEMENT 4 - MASKED FINE-TUNING

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms and loaders (same as before)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = OxfordIIITPet(root='./dataset', split='trainval', target_types='binary-category', transform=transform, download=True)
test_dataset = OxfordIIITPet(root='./dataset', split='test', target_types='binary-category', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Model setup
model = resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Masked fine-tuning setup
mask_fraction = 0.1  # Only update 10% of the parameters in model.fc
fc_weight = model.fc.weight
fc_bias = model.fc.bias

# Create random masks for weights and bias
weight_mask = torch.zeros_like(fc_weight, dtype=torch.bool)
bias_mask = torch.zeros_like(fc_bias, dtype=torch.bool)

# Randomly select a subset of parameters to update
num_weight_params = fc_weight.numel()
num_bias_params = fc_bias.numel()
weight_indices = np.random.choice(num_weight_params, int(mask_fraction * num_weight_params), replace=False)
bias_indices = np.random.choice(num_bias_params, int(mask_fraction * num_bias_params), replace=False)
weight_mask.view(-1)[weight_indices] = True
bias_mask.view(-1)[bias_indices] = True

# Optimizer and loss
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop with masking
def train_masked(model, loader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Apply mask to gradients
            with torch.no_grad():
                model.fc.weight.grad *= weight_mask
                model.fc.bias.grad *= bias_mask
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(loader):.4f}")

# Evaluation
def test(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Train and evaluate
print("Starting masked fine-tuning (random mask)...")
train_masked(model, train_loader, optimizer, criterion, num_epochs=5)
acc = test(model, test_loader)
print(f"Masked fine-tuning Test Accuracy: {acc:.2f}%")

# Baseline: full fine-tuning of fc layer (no mask)
model = resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
print("\nStarting baseline fine-tuning (all fc params trainable)...")
train_masked(model, train_loader, optimizer, criterion, num_epochs=5)  # No mask applied inside, so all params update
acc = test(model, test_loader)
print(f"Baseline fine-tuning Test Accuracy: {acc:.2f}%")