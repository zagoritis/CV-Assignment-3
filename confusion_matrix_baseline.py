import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from baseline import (
    JesterDataset2D,
    BaselineCNN,
    val_transform,
    data_root,
    val_csv,
    batch_size,
    num_workers,
    num_classes,
    label_to_idx,
    gesture_labels,
    num_workers,
    input_size,
    video_mean,
    video_std,
)

def get_predictions_and_labels(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            # inputs: (B, C, H, W) – already correct for BaselineCNN / ResNet18
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    return all_predictions, all_labels


def plot_confusion_matrix(conf_matrix, class_names, filename):
    plt.figure(figsize=(8, 8))
    im = plt.imshow(conf_matrix, interpolation="nearest")
    plt.title("Confusion Matrix for Baseline 2D Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)

    cbar = plt.colorbar(im, fraction=0.035, pad=0.04)
    cbar.ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    val_transform = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=video_mean, std=video_std),
    ])

    val_dataset = JesterDataset2D(
        csv_file=val_csv,
        root_dir=data_root,
        label_map=label_to_idx,
        transform=val_transform,
        train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load("jester_baseline_model.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions, labels = get_predictions_and_labels(model, val_loader, device)
    conf_matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    plot_confusion_matrix(conf_matrix, gesture_labels, filename="confusion_matrix_baseline.png")

if __name__ == "__main__":
    main()