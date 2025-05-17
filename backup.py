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