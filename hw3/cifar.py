import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np


#————device--------------------------------
#判断算法是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device} device")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#---dataset--------------------------------
# Load the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

#---model--------------------------------
# Network 1: Feed-Forward (fully connected only)
#我现在要创建一个自己的模型，这个模型要遵守 PyTorch 的规则，并且要实现前向传播和反向传播
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( #输入先进第 1 层，再进第 2 层，再进第 3 层……一直传下去。
            nn.Flatten(),
            nn.Linear(3072, 1024), nn.ReLU(),
            nn.Linear(1024,  512), nn.ReLU(),
            nn.Linear( 512,  256), nn.ReLU(),
            nn.Linear( 256,   10),
        )
    def forward(self, x):
        return self.net(x)


# Network 2: Small CNN
# Conv(3→32) → Pool → Conv(32→64) → Pool → FC 64*8*8→512 → FC→10
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 32→16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))



# Network 3: Deeper CNN with Dropout
# Conv(3→32) → Conv(32→64) → Pool → Conv(64→128) → Pool → Dropout → FC→256 → FC→10
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 32→16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

#  ── Training ─────────────────────────────────────────────────────────────────
def compute_train_loss(model, criterion):
    """Compute loss over the full training set (eval/no-grad mode)."""
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / total


def train_model(model, name, lr=1e-3, min_epochs=5, max_epochs=30):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    save_path = f"{name}.pth"

    epoch_losses = []
    prev_loss = float('inf')

    for epoch in range(1, max_epochs + 1):
        # ── mini-batch training pass ──
        model.train()
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        # ── full-set evaluation ──
        ep_loss = compute_train_loss(model, criterion)
        epoch_losses.append(ep_loss)
        print(f"[{name}] Epoch {epoch:2d}  train loss: {ep_loss:.4f}")

        # Save after every epoch
        torch.save(model.state_dict(), save_path)

        # Early stopping (only after min_epochs)
        if epoch > min_epochs and ep_loss > prev_loss:
            print(f"[{name}] Loss increased — stopping early at epoch {epoch}.")
            break
        prev_loss = ep_loss

    return epoch_losses


# ── Testing ───────────────────────────────────────────────────────────────────
def test_model(model, name):
    """Return accuracy and (correct_sample, incorrect_sample)."""
    model.eval()
    correct, total = 0, 0
    correct_ex = incorrect_ex = None  # (img_tensor, true_label, pred_label)

    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            for i in range(len(labels)):
                is_correct = (preds[i] == labels[i]).item()
                correct += is_correct
                total += 1
                img_cpu = imgs[i].cpu()
                t, p = labels[i].item(), preds[i].item()
                if is_correct and correct_ex is None:
                    correct_ex = (img_cpu, t, p)
                if not is_correct and incorrect_ex is None:
                    incorrect_ex = (img_cpu, t, p)

    acc = correct / total
    print(f"[{name}] Test accuracy: {acc*100:.2f}%")
    return acc, correct_ex, incorrect_ex


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_loss(losses, name):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f"{name} — Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{name}_loss.png", dpi=150)
    plt.close()
    print(f"Saved {name}_loss.png")


def unnormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std  = torch.tensor([0.2470, 0.2435, 0.2616])
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)


def show_examples(correct_ex, incorrect_ex, name):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, ex, title in zip(
        axes,
        [correct_ex, incorrect_ex],
        ["Correct", "Incorrect"]
    ):
        img = unnormalize(ex[0]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"{title}\nTrue: {CLASS_NAMES[ex[1]]}\nPred: {CLASS_NAMES[ex[2]]}", fontsize=9)
        ax.axis('off')
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"{name}_examples.png", dpi=150)
    plt.close()
    print(f"Saved {name}_examples.png")


def main():
    # ── Main ──────────────────────────────────────────────────────────────────
    networks = [
        (FeedForwardNet(), "Net1_FeedForward"),
        (SmallCNN(),       "Net2_SmallCNN"),
        (DeepCNN(),        "Net3_DeepCNN"),
    ]

    results = {}
    for model, name in networks:
        print(f"\n{'='*50}")
        print(f" Training {name}")
        print(f"{'='*50}")
        losses = train_model(model, name)
        plot_loss(losses, name)

        acc, correct_ex, incorrect_ex = test_model(model, name)
        show_examples(correct_ex, incorrect_ex, name)
        results[name] = acc

    print("\n\n=== Final Results ===")
    for name, acc in results.items():
        print(f"{name}: {acc*100:.2f}%")
    best = max(results, key=results.get)
    worst = min(results, key=results.get)
    print(f"\nBest:  {best}")
    print(f"Worst: {worst}")


if __name__ == "__main__":
    main()