# Import dependencies
import torch
import argparse
import time
import os
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale, Compose, Resize


# ── Helpers ──────────────────────────────────────────────────────────────
def get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    return torch.device(dev)


def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# ── Image Classifier Neural Network ─────────────────────────────────────
class ImageClassifier(nn.Module):
    """CNN for MNIST digit classification (1×28×28 → 10 classes)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)), #Extract simple features (edges, lines)
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)), #Extract more complex features (shapes, patterns)
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)), #even more complex features (textures, combinations of shapes)
            nn.ReLU(),
            nn.Flatten(), #convert it into long vector 
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x):
        return self.model(x)


# ── Evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataloader, device):
    """Compute accuracy and average loss on a dataset."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        running_loss += loss_fn(preds, y).item() * y.size(0)
        correct += (preds.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total, running_loss / total


# ── Prediction ───────────────────────────────────────────────────────────
def predict_image(model, image_path, device):
    """Load an image file and predict its digit class."""
    transform = Compose([
        Grayscale(num_output_channels=1),
        Resize((28, 28)),
        ToTensor(),
    ])
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence


# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="MNIST Digit Classifier — PyTorch CNN",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--model-path", type=str, default="model_state.pt",
                        help="Path to save/load model weights (default: model_state.pt)")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training; load existing weights and predict only")
    parser.add_argument("--predict", type=str, nargs="*",
                        help="Image file(s) to classify after training")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print("  MNIST Digit Classifier")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model path   : {args.model_path}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────
    train_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    print(f"  Train samples: {len(train_data):,}")
    print(f"  Test samples : {len(test_data):,}")
    print("=" * 60)

    # ── Model / Optimizer / Loss ─────────────────────────────────────
    clf = ImageClassifier().to(device)
    opt = Adam(clf.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in clf.parameters())
    print(f"  Parameters   : {param_count:,}")
    print("=" * 60)

    # ── Training ─────────────────────────────────────────────────────
    if not args.no_train:
        print("\n[TRAIN] Starting training ...\n")
        total_start = time.time()
        best_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            running_loss = 0.0
            batches = 0

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                yhat = clf(X)
                loss = loss_fn(yhat, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                running_loss += loss.item()
                batches += 1

            avg_loss = running_loss / batches
            epoch_time = time.time() - epoch_start

            # Evaluate on test set every epoch
            test_acc, test_loss = evaluate(clf, test_loader, device)

            marker = ""
            if test_acc > best_acc:
                best_acc = test_acc
                marker = " ★ best"

            print(
                f"  Epoch {epoch:>3}/{args.epochs}  │  "
                f"train_loss: {avg_loss:.4f}  │  "
                f"test_loss: {test_loss:.4f}  │  "
                f"test_acc: {test_acc * 100:5.2f}%  │  "
                f"time: {format_time(epoch_time)}{marker}"
            )

        total_time = time.time() - total_start
        print(f"\n[TRAIN] Finished in {format_time(total_time)}")
        print(f"[TRAIN] Best test accuracy: {best_acc * 100:.2f}%")

        # Save model
        with open(args.model_path, "wb") as f:
            save(clf.state_dict(), f)
        print(f"[SAVE]  Model saved to {args.model_path}")

    else:
        # ── Load existing weights ────────────────────────────────────
        if not os.path.exists(args.model_path):
            print(f"[ERROR] Model file not found: {args.model_path}")
            exit(1)
        with open(args.model_path, "rb") as f:
            clf.load_state_dict(load(f, map_location=device, weights_only=True))
        print(f"[LOAD]  Model loaded from {args.model_path}")

        # Quick test evaluation
        test_acc, test_loss = evaluate(clf, test_loader, device)
        print(f"[EVAL]  Test accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}")

    # ── Predict on images ────────────────────────────────────────────
    image_files = args.predict
    if image_files is None:
        # Default: predict on all img_*.jpg in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_files = sorted(
            os.path.join(script_dir, f)
            for f in os.listdir(script_dir)
            if f.startswith("img_") and f.endswith(".jpg")
        )

    if image_files:
        print("\n" + "=" * 60)
        print("  Predictions")
        print("=" * 60)
        for img_path in image_files:
            if not os.path.exists(img_path):
                print(f"  [SKIP] {img_path} — file not found")
                continue
            pred, conf = predict_image(clf, img_path, device)
            bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
            print(f"  {os.path.basename(img_path):>12s}  →  digit {pred}  |  {bar} {conf * 100:.1f}%")
        print("=" * 60)