import torch
import matplotlib.pyplot as plt

def plot_learning(baseline_stats, improved_stats):
    epochs_baseline = range(1, len(baseline_stats["train_loss"]) + 1)
    epochs_improved = range(1, len(improved_stats["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs_baseline, baseline_stats["train_acc"], label="Baseline train")
    plt.plot(epochs_baseline, baseline_stats["val_acc"], label="Baseline val")
    plt.plot(epochs_improved, improved_stats["train_acc"], label="Improved train")
    plt.plot(epochs_improved, improved_stats["val_acc"], label="Improved val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Train/Val Accuracy for each Epoch")
    plt.tight_layout()
    plt.savefig("accuracy_curves.png", dpi=300)

    plt.figure()
    plt.plot(epochs_baseline, baseline_stats["train_loss"], label="Baseline train")
    plt.plot(epochs_baseline, baseline_stats["val_loss"], label="Baseline val")
    plt.plot(epochs_improved, improved_stats["train_loss"], label="Improved train")
    plt.plot(epochs_improved, improved_stats["val_loss"], label="Improved val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train/Val Loss for each Epoch")
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=300)

def plot_accuracy(baseline_stats, improved_stats):
    baseline_acc = max(baseline_stats["val_acc"])
    improved_acc = max(improved_stats["val_acc"])

    models = ["Baseline 2D Model", "Improved 3D Model"]
    accs = [baseline_acc, improved_acc]

    plt.figure()
    plt.bar(models, accs)
    plt.ylabel("Validation Accuracy")
    plt.title("Final Validation Accuracy")
    plt.tight_layout()
    plt.savefig("final_accuracy.png", dpi=300)

def main():
    baseline_stats = torch.load("baseline_history.pt")
    improved_stats = torch.load("improved_history.pt")

    plot_learning(baseline_stats, improved_stats)
    plot_accuracy(baseline_stats, improved_stats)

if __name__ == "__main__":
    main()