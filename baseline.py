import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
use_small_train = True
data_root = "small-20bn-jester-v1"
train_csv = "jester-v1-small-train.csv" if use_small_train else "jester-v1-train.csv"
val_csv = "jester-v1-validation.csv"
labels_csv = "jester-v1-labels.csv"

num_epochs = 20
batch_size = 32          # you can lower this if you get OOM
learning_rate = 0.001
input_size = 112
num_workers = 8
dataset_repeat = 2       # repeat to have more iterations per epoch

# -----------------------------
# Labels
# -----------------------------
with open(labels_csv, 'r') as f:
    gesture_labels = [line.strip() for line in f.readlines()]
label_to_idx = {label: idx for idx, label in enumerate(gesture_labels)}
num_classes = len(gesture_labels)

# -----------------------------
# Dataset (2D single-frame)
# -----------------------------
class JesterDataset2D(Dataset):
    """
    Baseline dataset:
    - Reads all frames from a video folder.
    - For training: picks a random frame.
    - For validation: picks the middle frame.
    - Returns a single image tensor (C, H, W) and label.
    """
    def __init__(self, csv_file, root_dir, label_map, transform=None, train=True):
        self.root_dir = root_dir
        self.label_map = label_map
        self.transform = transform
        self.train = train
        self.repeat_factor = dataset_repeat
        self.samples = []
        
        with open(csv_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_id, label_name = line.split(';')
                if label_name not in self.label_map:
                    continue
                class_idx = self.label_map[label_name]
                self.samples.append((video_id, class_idx))
        # cache frame file lists for speed
        self.frame_cache = {}

    def __len__(self):
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, idx):
        base_idx = idx % len(self.samples)
        video_id, class_idx = self.samples[base_idx]
        video_folder = os.path.join(self.root_dir, str(video_id))
        
        if video_id not in self.frame_cache:
            frame_files = sorted(
                [f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))]
            )
            self.frame_cache[video_id] = frame_files
        frames_list = self.frame_cache[video_id]

        if len(frames_list) == 0:
            raise RuntimeError(f"No frames found in video folder {video_folder}")
        
        if self.train:
            # random frame during training
            frame_index = random.randint(0, len(frames_list) - 1)
        else:
            # middle frame for validation
            frame_index = len(frames_list) // 2

        frame_file = frames_list[frame_index]
        frame_path = os.path.join(video_folder, frame_file)
        image = Image.open(frame_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = class_idx
        return image, label

# -----------------------------
# Transforms (same mean/std as 3D model)
# -----------------------------
video_mean = [0.43216, 0.394666, 0.37645]
video_std = [0.22803, 0.22145, 0.216989]

train_transform = transforms.Compose([
    transforms.Resize((128, 171)),
    # Make position / scale less trivial
    transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=video_mean, std=video_std),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 171)),
    transforms.CenterCrop((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=video_mean, std=video_std),
])


# -----------------------------
# Baseline 2D model
# -----------------------------

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # No pretrained weights: starts from random init
        self.model = resnet18(weights=None)   # or resnet18() in older torchvision
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def forward(self, x):
        # x shape: (B, C, H, W) single frame
        return self.model(x)

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predictions = torch.max(logits, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

# -----------------------------
# Training loop
# -----------------------------
def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, scheduler):
    model = model.to(device)
    best_accuracy = 0.0
    scaler = GradScaler()
    
    stats = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch +1}/{num_epochs}', position=0, leave=True) as pbar:
            for inputs, labels in train_loader:
                # inputs: (B, C, H, W) – already correct for ResNet18
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        print('\n'+f'Training set: Average loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}')

        val_loss, val_acc  = evaluate(model, val_loader, criterion, device)
        print(f'Validation set: Average loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}')

        stats["train_loss"].append(train_loss)
        stats["train_acc"].append(train_acc)
        stats["val_loss"].append(val_loss)
        stats["val_acc"].append(val_acc)

        if best_accuracy < val_acc:
            best_accuracy = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, 'jester_improved_model.ckpt')

        if scheduler:
            scheduler.step()

    return stats

# -----------------------------
# Main
# -----------------------------
def main():
    train_dataset = JesterDataset2D(
        csv_file=train_csv,
        root_dir=data_root,
        label_map=label_to_idx,
        transform=train_transform,
        train=True,
    )
    val_dataset = JesterDataset2D(
        csv_file=val_csv,
        root_dir=data_root,
        label_map=label_to_idx,
        transform=val_transform,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = BaselineCNN(num_classes=num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    stats = train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, scheduler=scheduler)
    torch.save(stats, "baseline_stats.pt")

if __name__ == "__main__":
    main()
