import torch
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import random_split, Dataset

# Set device. Use GPU if available else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------- BINARY CLASSIFICATION ---------------------

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
optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # TUNE

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
start_time = time.time() # <--- RECORD START TIME
train_model(num_epochs=30) # TUNE
end_time = time.time() # <--- RECORD END TIME
duration = end_time - start_time
print(f"Training time: {duration/60:.2f} minutes")
test_model()



# --------------------- MULTI-CLASS CLASSIFICATION ---------------------

# Define transforms for multi-class
train_transform_multi = transforms.Compose([
    transforms.Resize(224), # DEFAULT
    transforms.CenterCrop(224), # DEFAULT
    #transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), # Randomly crop the image to 224x224 with a scale of 75% to 100%
    #transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally with 50% probability
    #transforms.RandomRotation(15), # Randomly rotate the image by up to +/- 15 degrees
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_multi = transforms.Compose([ # Minimal for test/val
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset wrapper to apply a specific transform
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

# Load dataset for multi-class breed classification
base_train_val_dataset = OxfordIIITPet(root='./dataset', split='trainval', target_types='category', download=True)
test_dataset_multi_raw = OxfordIIITPet(root='./dataset', split='test', target_types='category', download=True)

# Split training data for validation
num_train_val_samples = len(base_train_val_dataset)
val_fraction = 0.2 # 20% for validation
num_val_samples = int(val_fraction * num_train_val_samples)
num_train_samples_for_split = num_train_val_samples - num_val_samples

# These subsets will contain (PIL Image, label) tuples
train_subset_raw, val_subset_raw = random_split(base_train_val_dataset, [num_train_samples_for_split, num_val_samples])

# Now apply the correct transforms using the wrapper
train_subset_multi = TransformedDataset(train_subset_raw, transform=train_transform_multi)
val_subset_multi = TransformedDataset(val_subset_raw, transform=test_transform_multi) # Use test_transform_multi for validation
test_dataset_multi = TransformedDataset(test_dataset_multi_raw, transform=test_transform_multi)
print(f"Multi-class Dataset loaded. Training samples: {len(train_subset_multi)}, Validation samples: {len(val_subset_multi)}, Test samples: {len(test_dataset_multi)}")

# Create DataLoader objects for the actual data subsets. shuffle=True shuffles the data each epoch
actual_train_loader = DataLoader(train_subset_multi, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
actual_val_loader = DataLoader(val_subset_multi, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
actual_test_loader = DataLoader(test_dataset_multi, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)


# Training loop
def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for imgs, labels_batch in train_loader:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
def test_model(loader_to_use):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels_batch in loader_to_use: # Use the passed loader
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            outputs = model(imgs)
            # For multi-class, outputs.shape will be (batch_size, 37)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    accuracy = 100 * correct / total
    return accuracy


# --- Strategy 1: Fine-tune l layers simultaneously ---
print("\nStarting Strategy 1...")
strategy1_start_time = time.time() # <--- RECORD START TIME

num_epochs_strat1 = 15  # TUNE
lr_strat1 = 1e-5      # TUNE Often smaller when fine-tuning more layers

val_accuracies_per_l = {}
best_val_accuracy = 0.0
best_l_config = None

# ResNet18 layers: layer1, layer2, layer3, layer4.
# l_val = 0: only FC layer (already handled by default if no layers are unfrozen)
# l_val = 1: FC + layer4
# ...
# l_val = 4: FC + layer4 + layer3 + layer2 + layer1
max_l = 4 # For ResNet18, there are 4 main layer blocks

for l_val in range(1, max_l + 1): # l_val from 1 to max_l
    print(f"\n    STRATEGY 1: Training with FC + last {l_val} ResNet block(s) unfrozen")

    # Re-assign global `model`
    model = resnet18(weights='IMAGENET1K_V1')
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer (always trainable)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37) # model.fc.parameters() are requires_grad=True by default

    # Unfreeze the last l_val blocks
    if l_val >= 1: # Unfreeze layer4
        print("Unfreezing model.layer4")
        for param in model.layer4.parameters():
            param.requires_grad = True
    if l_val >= 2: # Unfreeze layer3
        print("Unfreezing model.layer3")
        for param in model.layer3.parameters():
            param.requires_grad = True
    if l_val >= 3: # Unfreeze layer2
        print("Unfreezing model.layer2")
        for param in model.layer2.parameters():
            param.requires_grad = True
    if l_val >= 4: # Unfreeze layer1
        print("Unfreezing model.layer1")
        for param in model.layer1.parameters():
            param.requires_grad = True
    
    model = model.to(device)

    # Re-assign global `criterion`
    criterion = nn.CrossEntropyLoss()

    # Re-assign global `optimizer`
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters for l={l_val}: {sum(p.numel() for p in params_to_optimize)}")
    optimizer = optim.Adam(params_to_optimize, lr=lr_strat1) # Without L2 regularization
    # optimizer = optim.Adam(params_to_optimize, lr=lr_strat1, weight_decay=1e-4) # With L2 regularization

    # Re-assign global `train_loader` for the train_model function
    train_loader = actual_train_loader

    # Call the redefined train_model function
    print(f"Starting training for l={l_val} and num_epochs={num_epochs_strat1}...")
    train_model(num_epochs=num_epochs_strat1) # Uses global vars

    # Perform validation
    print(f"Validating for l={l_val}...")
    current_val_accuracy = test_model(actual_val_loader) # Uses global model
    val_accuracies_per_l[l_val] = current_val_accuracy
    print(f"Validation Accuracy for l={l_val}: {current_val_accuracy:.4f}%")

    if current_val_accuracy > best_val_accuracy:
        best_val_accuracy = current_val_accuracy
        best_l_config = l_val
        # Save the best model's state_dict
        torch.save(model.state_dict(), f'best_model_strat1_l_{best_l_config}.pth')
        print(f"Saved new best model for l={best_l_config}")

strategy1_end_time = time.time() # <--- RECORD END TIME
strategy1_duration = strategy1_end_time - strategy1_start_time

# After the loop, evaluate the best configuration on the test set
print("\n--- Strategy 1 Finished ---")
if best_l_config is not None:
    print(f"Best l based on validation accuracy: {best_l_config} (Accuracy: {val_accuracies_per_l[best_l_config]:.4f}%)")
    print(f"Loading and evaluating best model (l={best_l_config}) on the Test Set...")
    
    # Re-setup the model architecture for the best_l_config
    model = resnet18(weights=None) # Initialize without pre-trained weights if loading all
    model.fc = nn.Linear(model.fc.in_features, 37) # Replace the final fully connected layer to match the saved model

    # Load the saved state dictionary for the best model
    model.load_state_dict(torch.load(f'best_model_strat1_l_{best_l_config}.pth', weights_only=True))
    model = model.to(device)
    
    final_test_accuracy = test_model(actual_test_loader) # Uses the loaded best model
    print(f"Final Test Accuracy (with best l={best_l_config} config): {final_test_accuracy:.4f}%")
else:
    print("No validation results to determine the best configuration for testing.")

    
print(f"Strategy 1 training time: {strategy1_duration/60:.2f} minutes")
