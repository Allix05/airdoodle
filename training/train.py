"""Train DoodleNet on the unified digits+letters+doodles dataset."""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from classes import CLASSES
from model import DoodleNet

DATA_PATH = "data/unified.npz"
CHECKPOINT_DIR = "checkpoints"
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
VAL_FRACTION = 0.08
MAX_SHIFT = 2  # pixels; cheap translation augmentation, whole batch shifted together


def random_shift_batch(xb, max_shift=MAX_SHIFT):
    """Shift a whole batch by the same random (dx, dy) in [-max_shift, max_shift].
    Zero-padded (matches the black background convention), fully vectorized --
    no per-sample Python loop, so it's nearly free on top of training time.
    """
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    if dx == 0 and dy == 0:
        return xb
    h, w = xb.shape[-2], xb.shape[-1]
    padded = F.pad(xb, (max_shift, max_shift, max_shift, max_shift))
    top = max_shift - dy
    left = max_shift - dx
    return padded[:, :, top : top + h, left : left + w]


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = np.load(DATA_PATH, allow_pickle=True)
    X, y = data["X"], data["y"]
    assert list(data["classes"]) == CLASSES

    X = X.astype(np.float32) / 255.0
    X = X[:, None, :, :]  # add channel dim -> (N, 1, 28, 28)

    n_val = int(len(X) * VAL_FRACTION)
    X_train, y_train = X[n_val:], y[n_val:]
    X_val, y_val = X[:n_val], y[:n_val]
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = DoodleNet(len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = random_shift_batch(xb)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
        val_acc = correct / len(val_ds)
        scheduler.step()

        print(f"[epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))

    print(f"Best val acc: {best_val_acc:.4f}")

    with open(os.path.join(CHECKPOINT_DIR, "classes.json"), "w") as f:
        json.dump(CLASSES, f)


if __name__ == "__main__":
    main()
