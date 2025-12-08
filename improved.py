import os
import random
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

use_small_train = True
data_root = "small-20bn-jester-v1"
train_csv = "jester-v1-small-train.csv" if use_small_train else "jester-v1-train.csv"
val_csv = "jester-v1-validation.csv"
labels_csv = "jester-v1-labels.csv"

num_epochs = 10
batch_size = 2
learning_rate = 0.001
input_size = 112
num_workers = 8
frames_per_clip = 32
dataset_repeat = 2

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
        self.num_frames = frames_per_clip
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
        self.frame_cache = {}

    def __len__(self):
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, idx):
        base_idx = idx % len(self.samples)
        video_id, class_idx = self.samples[base_idx]
        video_folder = os.path.join(self.root_dir, str(video_id))
        if video_id not in self.frame_cache:
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))])
            self.frame_cache[video_id] = frame_files
        frames_list = self.frame_cache[video_id]

        if self.train:
            if len(frames_list) == 0:
                raise RuntimeError(f"No frames found in video folder {video_folder}")
            max_start = max(0, len(frames_list) - self.num_frames)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [min(len(frames_list) - 1, start_idx + i) for i in range(self.num_frames)]
        else:
            if len(frames_list) == 0:
                raise RuntimeError(f"No frames found in video folder {video_folder}")
            max_start = max(0, len(frames_list) - self.num_frames)
            start_idx = max_start // 2 if max_start > 0 else 0
            frame_indices = [min(len(frames_list) - 1, start_idx + i) for i in range(self.num_frames)]

        while len(frame_indices) < self.num_frames and len(frame_indices) > 0:
            frame_indices.append(frame_indices[-1])

        frames = []
        for frame_index in frame_indices:
            frame_file = frames_list[frame_index]
            frame_path = os.path.join(video_folder, frame_file)
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        video = torch.stack(frames)
        label = class_idx
        return video, label

video_mean = [0.43216, 0.394666, 0.37645]
video_std = [0.22803, 0.22145, 0.216989]

transform = transforms.Compose([
    transforms.Resize((128, 171)),
    transforms.CenterCrop((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=video_mean, std=video_std)
])

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def evaluate(model, data_loader, criterion, device):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, labels in data_loader:
            inputs = inputs.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
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

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, scheduler):
    model = model.to(device)
    best_accuracy = 0.0
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch +1}/{num_epochs}', position=0, leave=True) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    logits = model(inputs)
                    loss = criterion(logits , labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        print('\n'+f'Training set: Average loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}')

        avg_loss, accuracy  = evaluate(model, val_loader, criterion, device)
        print(f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, 'jester_improved_model.ckpt')

        if scheduler:
            scheduler.step()

def main():
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

    model = ImprovedCNN(num_classes=num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, scheduler=scheduler)

if __name__ == "__main__":
    main()