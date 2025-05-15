import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import math

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. LoRA module for Linear layers
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0

    def forward(self, x):
        result = self.linear(x)
        if self.r > 0:
            # x: (batch, in_features)
            lora_out = self.dropout(x) @ self.lora_A.t()  # (batch, r)
            lora_out = lora_out @ self.lora_B.t()         # (batch, out_features)
            result = result + self.scaling * lora_out
        return result

# 3. Data transforms (add augmentation for train)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Load dataset and split train/val
full_train_dataset = OxfordIIITPet(root='./dataset', split='trainval', target_types='category', download=True)
test_dataset = OxfordIIITPet(root='./dataset', split='test', target_types='category', transform=test_transform, download=True)

val_fraction = 0.2
num_val = int(len(full_train_dataset) * val_fraction)
num_train = len(full_train_dataset) - num_val
train_subset, val_subset = random_split(full_train_dataset, [num_train, num_val])

# Apply transforms
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)

train_dataset = TransformedDataset(train_subset, train_transform)
val_dataset = TransformedDataset(val_subset, test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# 5. Model setup: LoRA + unfreeze last 2 blocks
model = resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
model.fc = LoRALinear(model.fc.in_features, 37, r=8, alpha=16, dropout=0.1)

# 6. Layer-wise learning rates
params = [
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
]
optimizer = optim.Adam(params)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

model = model.to(device)

# 7. Training with validation and checkpointing
def train(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=20):
    best_val_acc = 0
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        val_acc = test(model, val_loader)
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return best_val_acc

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

print("Starting improved ResNet18+LoRA fine-tuning...")
best_val_acc = train(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=20)
test_acc = test(model, test_loader)
print(f"Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy (best model): {test_acc:.2f}%")