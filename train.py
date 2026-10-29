"""
Satellite land-use classifier (EuroSAT RGB) - one script does everything:
data -> train (head, then full fine-tune) -> evaluate on unseen test set
-> figures + metrics -> ONNX export -> copy demo assets into web/.

Run:  python train.py
"""
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------- config
SEED = 42
IMG_SIZE = 224            # MobileNetV3 was pretrained at 224x224
BATCH = 64
HEAD_EPOCHS = 3           # stage 1: only the new classifier head trains
FT_EPOCHS = 10            # stage 2: all layers, lower LR (early stopping)
HEAD_LR, FT_LR = 1e-3, 1e-4
PATIENCE = 3              # early stopping on validation loss

ROOT = Path(__file__).parent
DATA, REPORTS, WEB = ROOT / "data", ROOT / "reports", ROOT / "web"
CKPT = ROOT / "best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet statistics - the pretrained backbone expects inputs normalised like this
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # satellite tiles have no "up"
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def run_epoch(model, dl, criterion, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given. Returns (loss, acc)."""
    model.train(optimizer is not None)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train_stage(name, model, train_dl, val_dl, history, state, epochs, lr, patience=None):
    """Train the currently trainable params with AdamW; save checkpoint on best val loss."""
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    bad = 0
    for ep in range(1, epochs + 1):
        t = time.time()
        tl, ta = run_epoch(model, train_dl, criterion, opt)
        vl, va = run_epoch(model, val_dl, criterion)
        for k, v in zip(history, (tl, vl, ta, va)):
            history[k].append(v)
        print(f"[{name}] epoch {ep}/{epochs}  train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f}  ({time.time() - t:.0f}s)", flush=True)
        if vl < state["best_val"]:
            state["best_val"], bad = vl, 0
            torch.save(model.state_dict(), CKPT)
        else:
            bad += 1
            if patience and bad >= patience:
                print(f"[{name}] early stopping")
                break


def save_figures(history, head_epochs, cm, classes, acc):
    """Training curves (loss + accuracy) and the test-set confusion matrix."""
    ep = range(1, len(history["val_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "acc")):
        ax.plot(ep, history[f"train_{key}"], "o-", label="train")
        ax.plot(ep, history[f"val_{key}"], "o-", label="validation")
        ax.axvline(head_epochs + 0.5, color="gray", ls="--", lw=1, label="start fine-tune")
        ax.set_xlabel("epoch")
        ax.set_title("Loss" if key == "loss" else "Accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "training_curves.png", dpi=120)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix - test set (accuracy {acc:.2%})")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=120)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    REPORTS.mkdir(exist_ok=True)
    (WEB / "samples").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ data
    # Same files, two views: augmented for training, plain for val/test.
    train_view = datasets.EuroSAT(DATA, transform=TRAIN_TF, download=True)
    eval_view = datasets.EuroSAT(DATA, transform=EVAL_TF)
    classes = train_view.classes
    labels = np.array(train_view.targets)

    # Stratified 70 / 15 / 15 split - class balance kept in every split.
    idx = np.arange(len(labels))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, stratify=labels[tmp_idx], random_state=SEED)
    print(f"{len(classes)} classes | train {len(train_idx)} | val {len(val_idx)} | "
          f"test {len(test_idx)} | device {DEVICE}")

    # Few workers: on Windows each one is a new process that reloads the CUDA DLLs.
    def loader(ds, ids, shuffle, workers=2):
        return DataLoader(Subset(ds, ids), batch_size=BATCH, shuffle=shuffle, num_workers=workers,
                          pin_memory=True, persistent_workers=workers > 0)
    train_dl = loader(train_view, train_idx, True)
    val_dl = loader(eval_view, val_idx, False)
    test_dl = loader(eval_view, test_idx, False, workers=0)  # used once, at the end

    # ------------------------------------------------------------ model
    # Pretrained MobileNetV3-Small (ImageNet); replace the last layer with a 10-class head.
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(DEVICE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    state = {"best_val": float("inf")}

    # Stage 1: freeze the backbone, train only the classifier head.
    for p in model.features.parameters():
        p.requires_grad = False
    train_stage("head", model, train_dl, val_dl, history, state, HEAD_EPOCHS, HEAD_LR)
    head_epochs = len(history["val_loss"])

    # Stage 2: unfreeze everything, fine-tune with a 10x lower LR + early stopping.
    for p in model.parameters():
        p.requires_grad = True
    train_stage("finetune", model, train_dl, val_dl, history, state, FT_EPOCHS, FT_LR, PATIENCE)

    # ------------------------------------------------------------ evaluate on the unseen test set
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            trues += y.tolist()
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    print(f"\nTEST accuracy {acc:.4f} | macro F1 {f1:.4f}")
    print(classification_report(trues, preds, target_names=classes, digits=4))
    (REPORTS / "metrics.json").write_text(json.dumps({
        "test_accuracy": acc, "macro_f1": f1, "n_test": len(trues), "classes": classes,
        "history": history,
        "report": classification_report(trues, preds, target_names=classes, output_dict=True),
    }, indent=2))
    save_figures(history, head_epochs, confusion_matrix(trues, preds), classes, acc)

    # ------------------------------------------------------------ ONNX export
    # Input: 1x3x224x224 normalised float tensor -> output: 10 logits.
    model.cpu().eval()
    torch.onnx.export(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), WEB / "model.onnx",
                      input_names=["input"], output_names=["logits"], opset_version=17,
                      dynamo=False)

    # ------------------------------------------------------------ demo assets for web/
    for f in ("training_curves.png", "confusion_matrix.png"):
        shutil.copy(REPORTS / f, WEB / f)
    # One unseen test image per class as clickable sample tiles.
    raw = datasets.EuroSAT(DATA)
    for c, name in enumerate(classes):
        i = next(i for i in test_idx if labels[i] == c)
        raw[i][0].save(WEB / "samples" / f"{name}.jpg", quality=95)
    (WEB / "meta.json").write_text(json.dumps({
        "classes": classes, "test_accuracy": acc, "macro_f1": f1, "n_test": len(trues),
        "mean": MEAN, "std": STD, "img_size": IMG_SIZE}, indent=2))
    print(f"Done. Model, figures and samples written to {WEB}")


if __name__ == "__main__":
    main()
