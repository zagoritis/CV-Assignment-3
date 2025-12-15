import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix
import csv
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
    all_video_ids = []

    with torch.no_grad():
        for inputs, labels in data_loader:
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
    cm = conf_matrix.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums!=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)

    ax.set_title("Confusion Matrix for Baseline 2D Model")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.04, fraction=0.046)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    cbar.set_ticks([0, 0.5, 1.0])

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

    items, samples, video_ids = len(val_dataset), len(val_dataset.samples), []
    for idx in range(items):
        sample_idx = idx % samples
        video_id, _ = val_dataset.samples[sample_idx]
        video_ids.append(video_id)
    wrong = np.where(predictions != labels)[0]

    with open("baseline_misclassified.csv", "w", newline="", encoding="utf-8") as f:
        baseline_misclassified = csv.writer(f)
        baseline_misclassified.writerow(["video_id", "true_label", "pred_label"])
        for i in wrong:
            baseline_misclassified.writerow([video_ids[i], gesture_labels[int(labels[i])], gesture_labels[int(predictions[i])]])

    conf_matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    plot_confusion_matrix(conf_matrix, gesture_labels, filename="confusion_matrix_baseline.png")

if __name__ == "__main__":
    main()