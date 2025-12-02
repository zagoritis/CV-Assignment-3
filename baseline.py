import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ----- Configuration -----
use_small_train = True  # Toggle between small training set and full training set
data_root = "small-20bn-jester-v1"  # Directory containing video frame folders
train_csv = "jester-v1-small-train.csv" if use_small_train else "jester-v1-train.csv"
val_csv = "jester-v1-validation.csv"
labels_csv = "jester-v1-labels.csv"

# Training hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 1e-3
input_size = 112  # Resize images to 112x112 resolution for the model
num_workers = 4   # Number of DataLoader worker processes for data loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Optimize GPU performance for fixed-size inputs

# Checkpoint configuration
checkpoint_path = "jester_baseline.pth"
resume_from_checkpoint = False  # Set to True to load from checkpoint if available

# ----- Gesture Labels Mapping -----
# Load gesture labels and create a mapping from label name to class index
with open(labels_csv, 'r') as f:
    gesture_labels = [line.strip() for line in f.readlines()]
label_to_idx = {label: idx for idx, label in enumerate(gesture_labels)}
num_classes = len(gesture_labels)  # should be 27 for Jester

# ----- Dataset Definition -----
class JesterDataset(Dataset):
    """Custom Dataset for Jester gesture recognition videos. 
    Each item is one video, represented by a randomly selected frame image and a label."""
    def __init__(self, csv_file, root_dir, label_map, transform=None, train=True):
        """
        Args:
            csv_file (str): Path to the CSV file with format "video_id;label_name".
            root_dir (str): Directory with all the video frame folders.
            label_map (dict): Mapping from label names to numeric class indices.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): If True, use random frame sampling for augmentation. 
                          If False (validation), use a consistent frame (middle frame).
        """
        self.root_dir = root_dir
        self.label_map = label_map
        self.transform = transform
        self.train = train

        # Read the CSV file and parse video IDs and labels
        self.samples = []
        with open(csv_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_id, label_name = line.split(';')
                if label_name not in self.label_map:
                    # Skip if label not recognized (should not happen if labels file is complete)
                    continue
                class_idx = self.label_map[label_name]
                self.samples.append((video_id, class_idx))

        # Optional: shuffle the samples list if needed (not necessary as DataLoader can shuffle)
        # if self.train:
        #     random.shuffle(self.samples)

        # Frame cache to avoid listing frames repeatedly (improves performance)
        self.frame_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, class_idx = self.samples[idx]
        video_folder = os.path.join(self.root_dir, str(video_id))
        # List frames in the video folder (cache the list for efficiency)
        if video_id not in self.frame_cache:
            # Get all image files in the folder. Assumes frames are named like '00001.jpg', '00002.jpg', etc.
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))])
            self.frame_cache[video_id] = frame_files
        frames_list = self.frame_cache[video_id]

        # Choose a frame: random for training, middle frame for validation (for consistency)
        if self.train:
            frame_file = random.choice(frames_list)
        else:
            if len(frames_list) == 0:
                raise RuntimeError(f"No frames found in video folder {video_folder}")
            frame_index = len(frames_list) // 2  # middle frame index
            frame_file = frames_list[frame_index]

        frame_path = os.path.join(video_folder, frame_file)
        # Open image
        image = Image.open(frame_path).convert('RGB')
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        label = class_idx
        return image, label

# Define image transformations: resize, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),  # resize to a fixed size
    transforms.ToTensor(),
    # Normalize images (here we use mean=0.5, std=0.5 for each channel to scale pixel values to [-1,1])
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create training and validation dataset instances
train_dataset = JesterDataset(csv_file=train_csv, root_dir=data_root, label_map=label_to_idx, 
                              transform=transform, train=True)
val_dataset = JesterDataset(csv_file=val_csv, root_dir=data_root, label_map=label_to_idx, 
                            transform=transform, train=False)

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers, pin_memory=True)

# ----- Model Definition -----
class SimpleCNN(nn.Module):
    """A simple 2D CNN for image classification of gesture frames."""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Define a simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Conv layer 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # downsample by 2

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Conv layer 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # downsample by 2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Conv layer 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # downsample by 2
        )
        # After three conv/pool layers, image spatial size is reduced.
        # Compute the size of the feature map after the conv layers to define the first FC layer:
        # If input_size is 112, after 3 pools (factor 2 each), output size = 112/(2^3) = 14.
        # So feature map dimension = 64 * 14 * 14.
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size // 8) * (input_size // 8), 128),  # 64*14*14 -> 128 (for input_size=112)
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# Initialize model, loss function, optimizer
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----- Optional: Load from Checkpoint -----
start_epoch = 0
if resume_from_checkpoint and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        # If we saved a dict with model and optimizer state
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)))
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
    else:
        # If we saved only the model state_dict
        model.load_state_dict(checkpoint)
    print(f"Resumed training from checkpoint: {checkpoint_path}, starting at epoch {start_epoch}")

# ----- Training Loop -----
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_correct = 0
    running_total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training accuracy
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
    train_accuracy = 100.0 * running_correct / running_total

    # Validation loop (no gradient)
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            # Compute predictions and accumulate accuracy
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_accuracy = 100.0 * val_correct / val_total

    # Print epoch results
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Accuracy: {train_accuracy:.2f}% , Validation Accuracy: {val_accuracy:.2f}%")

    # Save checkpoint after each epoch (could also save best model only based on val_accuracy)
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint_data, checkpoint_path)

# After training, save final model (state_dict)
torch.save(model.state_dict(), "jester_baseline_model.pth")
print("Training completed. Model saved to jester_baseline_model.pth")
