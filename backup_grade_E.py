


# --- Run Strategy 1 ---

# These transforms are defined globally above
actual_train_loader, actual_val_loader, actual_test_loader = setup_dataloaders(
    train_transform=train_transform_multi,
    test_transform=test_transform_multi
    # val_fraction=0.2, batch_size=32, num_workers=2, pin_memory=True # DEFAULT VALUES
)


# # --- TEMPORARY VISUALIZATION CODE TO CHECK AUGMENTATIONS ---

# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     # Denormalize if you normalized in your transforms
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# # Get a batch of training data
# try:
#     inputs, classes = next(iter(actual_train_loader))

#     # Make a grid from batch
#     out = torchvision.utils.make_grid(inputs[:4]) # Show first 4 images

#     imshow(out, title=[str(x.item()) for x in classes[:4]])
#     plt.show() # Keep the plot window open
#     print("Displayed a batch of augmented images. Check if they look transformed.")
#     print("Run this multiple times to see if random augmentations change the images.")

# except Exception as e:
#     print(f"Error during visualization: {e}")
#     print("Ensure matplotlib is installed and train_loader is not empty.")

# # --- END OF TEMPORARY VISUALIZATION CODE ---





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
lr_strat1 = 1e-4      # TUNE Often smaller when fine-tuning more layers

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
    # optimizer = optim.Adam(params_to_optimize, lr=lr_strat1, weight_decay=1e-4) # With L2 regularization TUNE

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
