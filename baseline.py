import os
import random
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

use_small_train = True
data_root = "small-20bn-jester-v1"
train_csv = "jester-v1-small-train.csv" if use_small_train else "jester-v1-train.csv"
val_csv = "jester-v1-validation.csv"
labels_csv = "jester-v1-labels.csv"

num_epochs = 10
batch_size = 32
learning_rate = 1e-3
input_size = 112
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

with open(labels_csv, 'r') as f:
    gesture_labels = [line.strip() for line in f.readlines()]
label_to_idx = {label: idx for idx, label in enumerate(gesture_labels)}
num_classes = len(gesture_labels)

class JesterDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_map, transform=None, train=True):
        self.root_dir = root_dir
        self.label_map = label_map
        self.transform = transform
        self.train = train
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
        self.frame_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, class_idx = self.samples[idx]
        video_folder = os.path.join(self.root_dir, str(video_id))
        if video_id not in self.frame_cache:
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))])
            self.frame_cache[video_id] = frame_files
        frames_list = self.frame_cache[video_id]

        if self.train:
            frame_file = random.choice(frames_list)
        else:
            if len(frames_list) == 0:
                raise RuntimeError(f"No frames found in video folder {video_folder}")
            frame_index = len(frames_list) // 2
            frame_file = frames_list[frame_index]

        frame_path = os.path.join(video_folder, frame_file)
        image = Image.open(frame_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = class_idx
        return image, label

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size // 8) * (input_size // 8), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_accuracy = 100.0 * running_correct / running_total if running_total > 0 else 0.0

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_accuracy *= 100.0

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% , "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

def main(args):
    train_dataset = JesterDataset(csv_file=train_csv, 
                                  root_dir=data_root, 
                                  label_map=label_to_idx, 
                                  transform=transform, 
                                  train=True)
    val_dataset = JesterDataset(csv_file=val_csv, 
                                root_dir=data_root, 
                                label_map=label_to_idx, 
                                transform=transform, 
                                train=False)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=num_epochs)

    torch.save(model.state_dict(), "jester_baseline_model.pth")
    print("Training completed. Model saved to jester_baseline_model.pth")

if __name__ == "__main__":
    main()