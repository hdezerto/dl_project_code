import torch
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import random_split, Dataset

import numpy as np
import matplotlib.pyplot as plt
import torchvision # For make_grid and denormalizing

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
optimizer = optim.Adam(model.fc.parameters(), lr=0.002) # TUNE

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
    #transforms.Resize(224), # DEFAULT
    #transforms.CenterCrop(224), # DEFAULT
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), # Randomly crop the image to 224x224 with a scale of 75% to 100%
    transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally with 50% probability
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


def setup_dataloaders(root_dir='./dataset', val_fraction=0.2, batch_size=32, num_workers=2, pin_memory=True,
                      train_transform=None, test_transform=None):
    """Loads data, splits, and creates DataLoaders."""
    # Load dataset for multi-class breed classification
    base_train_val_dataset = OxfordIIITPet(root=root_dir, split='trainval', target_types='category', download=True)
    test_dataset_multi_raw = OxfordIIITPet(root=root_dir, split='test', target_types='category', download=True)

    # Split training data for validation
    num_train_val_samples = len(base_train_val_dataset)
    num_val_samples = int(val_fraction * num_train_val_samples)
    num_train_samples_for_split = num_train_val_samples - num_val_samples

    # These subsets will contain (PIL Image, label) tuples
    train_subset_raw, val_subset_raw = random_split(base_train_val_dataset, [num_train_samples_for_split, num_val_samples])

    # Now apply the correct transforms using the wrapper
    train_subset_multi = TransformedDataset(train_subset_raw, transform=train_transform)
    val_subset_multi = TransformedDataset(val_subset_raw, transform=test_transform)
    test_dataset_multi = TransformedDataset(test_dataset_multi_raw, transform=test_transform)
    
    print(f"Multi-class Dataset loaded. Training samples: {len(train_subset_multi)}, Validation samples: {len(val_subset_multi)}, Test samples: {len(test_dataset_multi)}")

    # Create DataLoader objects for the actual data subsets. shuffle=True shuffles the data each epoch
    train_loader = DataLoader(train_subset_multi, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset_multi, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset_multi, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device, batchnorm_mode="default"):
    """Trains the model for one epoch."""
    model.train()

    # Apply BatchNorm behavior based on the mode
    if batchnorm_mode == "freeze_params":
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False  # Freeze gamma and beta
    elif batchnorm_mode == "freeze_stats":
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # Freeze running_mean and running_var
    elif batchnorm_mode == "default":
        pass  # No need to explicitly set anything; rely on PyTorch's default behavior

    running_loss = 0.0
    for imgs, labels_batch in train_loader:
        imgs, labels_batch = imgs.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_epoch_loss = running_loss / len(train_loader)
    return avg_epoch_loss


def evaluate_model(model, loader_to_use, criterion, device):
    """Evaluates the model on a given loader."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels_batch in loader_to_use:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            outputs = model(imgs)
            if criterion: # Calculate loss if criterion is provided
                loss = criterion(outputs, labels_batch)
                running_loss += loss.item()
            # For multi-class, outputs.shape will be (batch_size, 37)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader_to_use) if criterion and len(loader_to_use) > 0 else 0.0
    return accuracy, avg_loss



# --- Strategy 1 ---

def run_fine_tuning_strategy_1(num_epochs, lr_fc, lr_backbone, device,
                               train_loader, val_loader, test_loader,
                               num_classes=37, model_save_prefix="strategy1_best_model",
                               factor=0.1, patience=2, l2_lambda=0.0, batchnorm_mode="default"):
    """
    Implements Strategy 1: Fine-tune l layers simultaneously with different LRs.
    """
    print("\nStarting Fine-Tuning Strategy 1...")
    start_time = time.time()

    val_accuracies_per_l = {}
    best_overall_val_accuracy = 0.0
    best_l_config_for_strategy = None
    
    max_l = 4 # For ResNet18

    for l_val in range(1, max_l + 1):
        print(f"\n    STRATEGY 1: Training with FC + last {l_val} ResNet block(s) unfrozen")

        # Initialize model for current l_val
        current_model = resnet18(weights='IMAGENET1K_V1')
        # Freeze all parameters initially
        for param in current_model.parameters():
            param.requires_grad = False
            # Replace the final fully connected layer (always trainable). model.fc.parameters() are requires_grad=True by default
        current_model.fc = nn.Linear(current_model.fc.in_features, num_classes)
        
        # Unfreeze layers
        current_backbone_params = []
        if l_val >= 1: # Unfreeze layer4
            print("    Unfreezing model.layer4")
            for param in current_model.layer4.parameters():
                param.requires_grad = True
                current_backbone_params.append(param)
        if l_val >= 2: # Unfreeze layer3
            print("    Unfreezing model.layer3")
            for param in current_model.layer3.parameters():
                param.requires_grad = True
                current_backbone_params.append(param)
        if l_val >= 3: # Unfreeze layer2
            print("    Unfreezing model.layer2")
            for param in current_model.layer2.parameters():
                param.requires_grad = True
                current_backbone_params.append(param)
        if l_val >= 4: # Unfreeze layer1
            print("    Unfreezing model.layer1")
            for param in current_model.layer1.parameters():
                param.requires_grad = True
                current_backbone_params.append(param)
        
        current_model = current_model.to(device)
        
        # Optimizer and Criterion for current l_val
        criterion = nn.CrossEntropyLoss()
        
        optimizer_grouped_parameters = [{'params': current_model.fc.parameters(), 'lr': lr_fc}]
        if current_backbone_params:
            optimizer_grouped_parameters.append({'params': current_backbone_params, 'lr': lr_backbone})
        
        current_optimizer = optim.Adam(optimizer_grouped_parameters, weight_decay=l2_lambda)
       
        # --- Initialize ReduceLROnPlateau Scheduler ---
        # mode='max' for accuracy, 'min' for loss.
        # factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
        # patience: Number of epochs with no improvement after which learning rate will be reduced.
        # verbose=True: Prints a message when the learning rate is reduced.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(current_optimizer, mode='max', factor=factor, patience=patience) # TUNE

        total_trainable_params_in_current_model = 0
        # Iterate over all parameters of the current_model being used for this l_val
        for param in current_model.parameters():
            if param.requires_grad:
                total_trainable_params_in_current_model += param.numel()
        print(f"    Number of trainable parameters for l={l_val}: {total_trainable_params_in_current_model}")

        print(f"    Starting training for l={l_val}, epochs={num_epochs}...")

        # Initialize a variable to track the previous learning rates
        previous_lrs = None
        for epoch in range(num_epochs):
            avg_train_loss = train_one_epoch(current_model, train_loader, current_optimizer, criterion, device, batchnorm_mode=batchnorm_mode)
            
            # Perform validation within the epoch loop for ReduceLROnPlateau
            epoch_val_accuracy, epoch_val_loss = evaluate_model(current_model, val_loader, criterion, device)
            
            # Step the ReduceLROnPlateau scheduler with the validation accuracy
            scheduler.step(epoch_val_accuracy)
            
            # Get the current learning rates
            current_lrs = [group['lr'] for group in current_optimizer.param_groups]

            # Print epoch details
            print(f"    l={l_val}, Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%, Val Loss: {epoch_val_loss:.4f}")
            
            # Print the learning rates if they have changed or if this is the first epoch
            if previous_lrs is None or current_lrs != previous_lrs:
                print(f"    Learning rates updated: {current_lrs}")
                previous_lrs = current_lrs  # Update the previous learning rates

        # Perform validation for current l_val
        print(f"    Validating for l={l_val}...")
        current_l_val_accuracy, current_l_val_loss = evaluate_model(current_model, val_loader, criterion, device)
        val_accuracies_per_l[l_val] = current_l_val_accuracy
        print(f"    Validation Accuracy for l={l_val}: {current_l_val_accuracy:.4f}%, Val Loss: {current_l_val_loss:.4f}")

        if current_l_val_accuracy > best_overall_val_accuracy:
            best_overall_val_accuracy = current_l_val_accuracy
            best_l_config_for_strategy = l_val
            # Save the best model's state_dict
            torch.save(current_model.state_dict(), f'{model_save_prefix}_l_{best_l_config_for_strategy}.pth')
            print(f"    Saved new best model (l={best_l_config_for_strategy})")

    end_time = time.time()
    strategy_duration_minutes = (end_time - start_time) / 60
    
    # After the loop, evaluate the best configuration on the test set
    final_test_accuracy_for_strategy = 0.0
    if best_l_config_for_strategy is not None:
        print(f"\n--- Strategy 1 Evaluation ---")
        print(f"Best l based on validation accuracy: {best_l_config_for_strategy} (Val Acc: {val_accuracies_per_l[best_l_config_for_strategy]:.4f}%)")
        print(f"Loading and evaluating best model (l={best_l_config_for_strategy}) on the Test Set...")
        
        # Re-setup the model architecture for the best_l_config
        best_model_for_strategy = resnet18(weights=None) # Initialize without pre-trained weights if loading all
        best_model_for_strategy.fc = nn.Linear(best_model_for_strategy.fc.in_features, num_classes) # Replace the final fully connected layer to match the saved model
        best_model_for_strategy.load_state_dict(torch.load(f'{model_save_prefix}_l_{best_l_config_for_strategy}.pth', weights_only=True)) # Load the saved state dictionary for the best model
        best_model_for_strategy = best_model_for_strategy.to(device)
        
        criterion_for_eval = nn.CrossEntropyLoss() # Re-init criterion for safety or pass it
        final_test_accuracy_for_strategy, _ = evaluate_model(best_model_for_strategy, test_loader, criterion_for_eval, device)
        print(f"Final Test Accuracy (best l={best_l_config_for_strategy}): {final_test_accuracy_for_strategy:.4f}%")
    else:
        print("No best configuration found for Strategy 1.")

    print(f"Training time: {strategy_duration_minutes:.2f} minutes")


# --- Run Strategy 1 ---

# These transforms are defined globally above
actual_train_loader, actual_val_loader, actual_test_loader = setup_dataloaders(
    train_transform=train_transform_multi,
    test_transform=test_transform_multi,
    batch_size=64 # TUNE
)


num_epochs_s1 = 15  # TUNE
# Learning rates for FC and backbone
lr_fc_s1 = 1e-3       # TUNE
lr_backbone_s1 = 1e-5 # TUNE
# Learning rate decay factor and patience for ReduceLROnPlateau
factor = 0.1 # TUNE
patience = 1 # TUNE
l2_lambda = 0.0 # L2 regularization TUNE

run_fine_tuning_strategy_1(
    num_epochs=num_epochs_s1,
    lr_fc=lr_fc_s1,
    lr_backbone=lr_backbone_s1,
    device=device,
    train_loader=actual_train_loader,
    val_loader=actual_val_loader,
    test_loader=actual_test_loader,
    num_classes=37,
    model_save_prefix="strategy1_best_model",
    factor=factor, patience=patience, l2_lambda=l2_lambda,
    batchnorm_mode="default" # TUNE ("freeze_params", "freeze_stats", "default")
)




# --- Strategy 2 ---

def run_fine_tuning_strategy_2(lr_fc, lr_backbone, device,
                               train_loader, val_loader, test_loader,
                               num_classes=37, model_save_prefix="strategy2_best_model",
                               factor=0.1, patience=2, l2_lambda=0.0, batchnorm_mode="default",
                               unfreeze_schedule=None):
    """
    Implements Strategy 2: Gradual unfreezing of layers during fine-tuning.
    """
    print("\nStarting Fine-Tuning Strategy 2...")
    start_time = time.time()

    val_accuracies_per_stage = {}
    best_overall_val_accuracy = 0.0
    best_stage_config = None

    # Initialize model
    current_model = resnet18(weights='IMAGENET1K_V1')
    # Freeze all parameters initially
    for param in current_model.parameters():
        param.requires_grad = False
    current_model.fc = nn.Linear(current_model.fc.in_features, num_classes)
    current_model = current_model.to(device)

    # Optimizer and Criterion
    criterion = nn.CrossEntropyLoss()

    # Default unfreeze schedule if none is provided
    if unfreeze_schedule is None:
        unfreeze_schedule = [
            {"layers_to_unfreeze": ["layer4"], "epochs": 5},
            {"layers_to_unfreeze": ["layer3"], "epochs": 5},
            {"layers_to_unfreeze": ["layer2"], "epochs": 5},
            {"layers_to_unfreeze": ["layer1"], "epochs": 5},
        ]

    total_epochs = sum(stage["epochs"] for stage in unfreeze_schedule)
    current_epoch = 0

    for stage_idx, stage in enumerate(unfreeze_schedule):
        layers_to_unfreeze = stage["layers_to_unfreeze"]
        stage_epochs = stage["epochs"]

        print(f"\nStage {stage_idx + 1}: Unfreezing layers {layers_to_unfreeze} for {stage_epochs} epochs...")

        # Unfreeze specified layers
        current_backbone_params = []
        for layer_name in layers_to_unfreeze:
            layer = getattr(current_model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
                current_backbone_params.append(param)

        # Define optimizer with updated parameter groups
        optimizer_grouped_parameters = [{'params': current_model.fc.parameters(), 'lr': lr_fc}]
        if current_backbone_params:
            optimizer_grouped_parameters.append({'params': current_backbone_params, 'lr': lr_backbone})
        current_optimizer = optim.Adam(optimizer_grouped_parameters, weight_decay=l2_lambda)

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(current_optimizer, mode='max', factor=factor, patience=patience)

        # Initialize a variable to track the previous learning rates
        previous_lrs = None
        
        # Train for the specified number of epochs in this stage
        for epoch in range(stage_epochs):
            current_epoch += 1
            avg_train_loss = train_one_epoch(current_model, train_loader, current_optimizer, criterion, device, batchnorm_mode=batchnorm_mode)
        
            # Perform validation
            epoch_val_accuracy, epoch_val_loss = evaluate_model(current_model, val_loader, criterion, device)
            scheduler.step(epoch_val_accuracy)
        
            # Get the current learning rates
            current_lrs = [group['lr'] for group in current_optimizer.param_groups]
        
            # Print epoch details
            print(f"    Epoch {current_epoch}/{total_epochs} (Stage {stage_idx + 1}), Train Loss: {avg_train_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%, Val Loss: {epoch_val_loss:.4f}")
        
            # Print the learning rates only if they have changed or if this is the first epoch
            if previous_lrs is None or current_lrs != previous_lrs:
                print(f"    Learning rates updated: {current_lrs}")
                previous_lrs = current_lrs  # Update the previous learning rates

        # Validate after the stage
        print(f"\nValidating after Stage {stage_idx + 1}...")
        stage_val_accuracy, stage_val_loss = evaluate_model(current_model, val_loader, criterion, device)
        val_accuracies_per_stage[stage_idx + 1] = stage_val_accuracy
        print(f"Validation Accuracy after Stage {stage_idx + 1}: {stage_val_accuracy:.4f}%, Val Loss: {stage_val_loss:.4f}")

        # Save the best model
        if stage_val_accuracy > best_overall_val_accuracy:
            best_overall_val_accuracy = stage_val_accuracy
            best_stage_config = stage_idx + 1
            torch.save(current_model.state_dict(), f'{model_save_prefix}_stage_{best_stage_config}.pth')
            print(f"    Saved new best model (Stage {best_stage_config})")
    
    end_time = time.time()

    # Evaluate the best model on the test set
    final_test_accuracy_for_strategy = 0.0
    if best_stage_config is not None:
        print(f"\n--- Strategy 2 Evaluation ---")
        print(f"Best stage based on validation accuracy: {best_stage_config} (Val Acc: {val_accuracies_per_stage[best_stage_config]:.4f}%)")
        print(f"Loading and evaluating best model (Stage {best_stage_config}) on the Test Set...")

        # Re-setup the model architecture
        best_model_for_strategy = resnet18(weights=None)
        best_model_for_strategy.fc = nn.Linear(best_model_for_strategy.fc.in_features, num_classes)
        best_model_for_strategy.load_state_dict(torch.load(f'{model_save_prefix}_stage_{best_stage_config}.pth', weights_only=True))
        best_model_for_strategy = best_model_for_strategy.to(device)

        criterion_for_eval = nn.CrossEntropyLoss()
        final_test_accuracy_for_strategy, _ = evaluate_model(best_model_for_strategy, test_loader, criterion_for_eval, device)
        print(f"Final Test Accuracy (best stage={best_stage_config}): {final_test_accuracy_for_strategy:.4f}%")
    else:
        print("No best configuration found for Strategy 2.")

    strategy_duration_minutes = (end_time - start_time) / 60
    print(f"Training time: {strategy_duration_minutes:.2f} minutes")



# --- Run Strategy 2 ---

# These transforms are defined globally above
actual_train_loader, actual_val_loader, actual_test_loader = setup_dataloaders(
    train_transform=train_transform_multi,
    test_transform=test_transform_multi,
    batch_size=64 # TUNE
)


unfreeze_schedule = [
    {"layers_to_unfreeze": ["layer4"], "epochs": 5}, # First stage unfreezes layer4 and trains for 5 epochs
    {"layers_to_unfreeze": ["layer3"], "epochs": 5}, # etc
    {"layers_to_unfreeze": ["layer2"], "epochs": 5},
    {"layers_to_unfreeze": ["layer1"], "epochs": 5},
]


run_fine_tuning_strategy_2(
    lr_fc=1e-3, # TUNE
    lr_backbone=1e-5, # TUNE
    device=device,
    train_loader=actual_train_loader,
    val_loader=actual_val_loader,
    test_loader=actual_test_loader,
    num_classes=37,
    model_save_prefix="strategy2_best_model",
    factor=0.1,
    patience=1,
    l2_lambda=1e-4, # TUNE 0.0
    batchnorm_mode="default",
    unfreeze_schedule=unfreeze_schedule
)




# --------------------- IMBALANCED CLASSES ---------------------

from collections import Counter
from torch.utils.data import Subset, WeightedRandomSampler

# Step 1: Simulate Imbalanced Classes for Cat Breeds
def create_imbalanced_dataset_for_cats(dataset, reduction_factor=0.2, cat_breed_indices=None):
    """
    Reduces the number of samples for each cat breed to simulate imbalance.
    Args:
        dataset: The original dataset (e.g., train_dataset).
        reduction_factor: Fraction of samples to keep for each cat breed (e.g., 0.2 for 20%).
        cat_breed_indices: List of indices corresponding to cat breeds in the dataset.
    Returns:
        A subset of the dataset with reduced samples for cat breeds and full samples for dog breeds.
    """
    class_counts = Counter([label for _, label in dataset])  # Count samples per class
    class_indices = {cls: [] for cls in class_counts.keys()}  # Store indices for each class

    # Group indices by class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Reduce samples for cat breeds
    reduced_indices = []
    for cls, indices in class_indices.items():
        if cls in cat_breed_indices:  # Apply imbalance only to cat breeds
            reduced_count = int(len(indices) * reduction_factor)
            reduced_indices.extend(indices[:reduced_count])  # Keep only a fraction of the samples
        else:  # Keep all samples for dog breeds
            reduced_indices.extend(indices)

    print(f"Original dataset size: {len(dataset)}, Reduced dataset size: {len(reduced_indices)}")
    return Subset(dataset, reduced_indices)

# Define cat breed indices (assuming first 19 classes are cat breeds)
cat_breed_indices = list(range(19))  # Adjust based on dataset class ordering

# Apply imbalance to the training dataset
imbalanced_train_dataset = create_imbalanced_dataset_for_cats(
    train_dataset_multi_raw, reduction_factor=0.2, cat_breed_indices=cat_breed_indices
)
imbalanced_train_loader = DataLoader(imbalanced_train_dataset, batch_size=64, shuffle=True)

# Step 2: Train with Normal Cross-Entropy Loss
print("\n--- Training with Normal Cross-Entropy Loss ---")
model = resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 37)  # Adjust for multi-class classification
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Normal cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train and evaluate
train_model(num_epochs=15)  # Reuse the train_model function
test_model()  # Reuse the test_model function

# Step 3: Weighted Cross-Entropy Loss
def compute_class_weights(dataset, cat_breed_indices):
    """
    Computes class weights based on the inverse frequency of each class.
    Args:
        dataset: The dataset (e.g., imbalanced_train_dataset).
        cat_breed_indices: List of indices corresponding to cat breeds in the dataset.
    Returns:
        A tensor of class weights.
    """
    class_counts = Counter([label for _, label in dataset])
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    return torch.tensor([class_weights[cls] for cls in sorted(class_weights.keys())], dtype=torch.float)

# Compute class weights
class_weights = compute_class_weights(imbalanced_train_dataset, cat_breed_indices)
print(f"Class weights: {class_weights}")

# Use weighted cross-entropy loss
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Train with weighted loss
print("\n--- Training with Weighted Cross-Entropy Loss ---")
model = resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 37)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_model(num_epochs=15)  # Reuse the train_model function
test_model()  # Reuse the test_model function

# Step 4: Over-Sampling Minority Classes
def create_sampler(dataset, cat_breed_indices):
    """
    Creates a sampler that over-samples minority classes.
    Args:
        dataset: The dataset (e.g., imbalanced_train_dataset).
        cat_breed_indices: List of indices corresponding to cat breeds in the dataset.
    Returns:
        A WeightedRandomSampler for the dataset.
    """
    class_counts = Counter([label for _, label in dataset])
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in dataset]
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

# Create sampler and DataLoader
sampler = create_sampler(imbalanced_train_dataset, cat_breed_indices)
oversampled_train_loader = DataLoader(imbalanced_train_dataset, batch_size=64, sampler=sampler)

# Train with over-sampling
print("\n--- Training with Over-Sampling ---")
model = resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 37)
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Normal cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_model(num_epochs=15)  # Reuse the train_model function
test_model()  # Reuse the test_model function

# Step 5: Evaluate Per-Class Accuracy
def evaluate_per_class_accuracy(model, loader, num_classes, device):
    """
    Evaluates per-class accuracy.
    Args:
        model: The trained model.
        loader: DataLoader for evaluation.
        num_classes: Total number of classes.
        device: Device (CPU or GPU).
    Returns:
        A dictionary with per-class accuracy.
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    per_class_accuracy = {cls: 100 * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
                          for cls in range(num_classes)}
    return per_class_accuracy

# Evaluate per-class accuracy
print("\n--- Evaluating Per-Class Accuracy ---")
per_class_accuracy = evaluate_per_class_accuracy(model, test_loader, num_classes=37, device=device)
print(f"Per-Class Accuracy: {per_class_accuracy}")