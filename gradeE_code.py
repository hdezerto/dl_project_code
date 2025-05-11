import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Set device. Use GPU if available else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Transform: Resize, normalize (ImageNet mean/std)
transform = transforms.Compose([
    transforms.Resize(224),        # Resize shortest side to 224, keep aspect ratio
    transforms.CenterCrop(224),    # Crop from the center to 224x224
    transforms.ToTensor(), # Convert to tensor
    # Normalize with ImageNet mean and std (check https://pytorch.org/hub/pytorch_vision_resnet/)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset as binary classification problem (cat vs dog). The resulting object is a list of tuples (image, label)
train_dataset = OxfordIIITPet(root='./dataset', split='trainval', target_types='binary-category', transform=transform, download=True)
test_dataset = OxfordIIITPet(root='./dataset', split='test', target_types='binary-category', transform=transform, download=True)
print("Dataset loaded. Number of training samples:", len(train_dataset), "Number of test samples:", len(test_dataset))

# Create DataLoader objects to efficiently load data in batches.
# - train_loader: loads training data in batches of 32 and shuffles the data each epoch (improves generalization).
# - test_loader: loads test data in batches of 32 without shuffling (for consistent evaluation).
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load a ResNet18 model pre-trained on ImageNet
model = resnet18(weights='IMAGENET1K_V1')
print("ResNet18 model loaded.")

# Freeze all the parameters in the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer to output 2 classes (cat vs dog)
model.fc = nn.Linear(model.fc.in_features, 2)
# Move the model to the selected device (GPU if available, else CPU)
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.006)

# Training loop
def train_model(num_epochs):
    model.train()  # Set model to training mode (enables dropout, batchnorm updates)
    for epoch in range(num_epochs):
        running_loss = 0  # Accumulate loss for this epoch
        for imgs, labels in train_loader:  # Loop over each batch in the training data
            imgs, labels = imgs.to(device), labels.to(device)  # Move data to GPU or CPU

            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(imgs)  # Forward pass: compute model predictions
            loss = criterion(outputs, labels)  # Compute loss between predictions and true labels
            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Add batch loss to epoch total
        # Print average loss for this epoch
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
def test_model():
    model.eval()  # Set model to evaluation mode (disables dropout, uses running stats for batchnorm)
    correct = total = 0
    with torch.no_grad():  # Disable gradient computation for efficiency during evaluation
        for imgs, labels in test_loader:  # Iterate over the test dataset in batches
            imgs, labels = imgs.to(device), labels.to(device)  # Move data to the appropriate device (CPU or GPU)
            outputs = model(imgs)  # Get model predictions (logits) for the batch. shape: (batch_size, 2)
            _, preds = torch.max(outputs, 1)  # Get the predicted class (index of max logit) for each sample
            correct += (preds == labels).sum().item()  # Count how many predictions are correct in this batch
            total += labels.size(0)  # Update the total number of samples seen so far
    print(f"Test Accuracy: {100 * correct / total:.4f}%")


# Run training and testing
train_model(num_epochs=10)
test_model()




