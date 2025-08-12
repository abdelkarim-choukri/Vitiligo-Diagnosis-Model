# scripts/train.py

import os
import argparse
import yaml
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# project imports
from scripts.dataset import VitiligoDataset
from scripts.model_classification import DualEfficientNetClassifier
from scripts.model_segmentation import DualInputSegmentationUNet


# ----------------------------- utils -----------------------------

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------------- training -----------------------------

def train_one_epoch(
    clf: nn.Module,
    seg: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    crit_cls,
    crit_seg,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    seg_w: float = 1.0,
    amp_dtype=torch.float16,
):
    clf.train()
    if seg is not None:
        seg.train()

    running = 0.0
    for cli, woo, mask, label in tqdm(loader, desc="Train", leave=False):
        cli, woo = cli.to(device, non_blocking=True), woo.to(device, non_blocking=True)
        label = label.float().to(device, non_blocking=True)
        if seg is not None:
            mask = mask.to(device, non_blocking=True).float()

        optim.zero_grad(set_to_none=True)

        with autocast(dtype=amp_dtype):
            logits = clf(cli, woo).squeeze(1)          # [B]
            loss = crit_cls(logits, label)

            if seg is not None:
                mask_logits = seg(cli, woo)            # [B,H,W]
                loss_seg = crit_seg(mask_logits, mask)
                loss = loss + seg_w * loss_seg

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running += loss.item()

    return running / max(1, len(loader))


@torch.no_grad()
def validate(
    clf: nn.Module,
    seg: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    crit_seg,
):
    clf.eval()
    if seg is not None:
        seg.eval()

    all_labels, all_probs, all_preds = [], [], []
    ious = []

    for cli, woo, mask, label in tqdm(loader, desc="Val", leave=False):
        cli, woo = cli.to(device, non_blocking=True), woo.to(device, non_blocking=True)

        # classification
        logits = clf(cli, woo).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_labels.extend(label.numpy().tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

        # segmentation IoU if available
        if seg is not None:
            mask = mask.to(device).float()
            mask_logits = seg(cli, woo)
            mp = (torch.sigmoid(mask_logits) >= 0.5).float()
            mt = mask
            inter = (mp * mt).sum(dim=(1, 2))
            union = ((mp + mt) > 0).float().sum(dim=(1, 2))
            batch_iou = torch.where(union == 0, (inter == 0).float(), inter / union).cpu().numpy()
            ious.extend(batch_iou.tolist())

    # metrics
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    mean_iou  = float(np.mean(ious)) if len(ious) else None

    metrics = dict(
        accuracy=acc,
        auc=auc,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # device & speed knobs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # TF32 on Ada/4090
    except Exception:
        pass

    # paths
    model_dir   = cfg["paths"]["model_dir"]
    results_dir = cfg["paths"]["results_dir"]
    ensure_dir(model_dir); ensure_dir(results_dir)

    # data
    train_ds = VitiligoDataset(
        csv_file=cfg["data"]["train_csv"],
        clinical_dir=cfg["data"]["clinical_dir"],
        wood_dir=cfg["data"]["wood_dir"],
        mask_dir=cfg["data"].get("mask_dir", None),
        transform=None,
        image_size=cfg["data"]["image_size"],
    )
    val_ds = VitiligoDataset(
        csv_file=cfg["data"]["val_csv"],
        clinical_dir=cfg["data"]["clinical_dir"],
        wood_dir=cfg["data"]["wood_dir"],
        mask_dir=cfg["data"].get("mask_dir", None),
        transform=None,
        image_size=cfg["data"]["image_size"],
    )

    num_workers = int(os.environ.get("NUM_WORKERS", "8"))
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    # models
    clf = DualEfficientNetClassifier(
        use_cbam=cfg["model"].get("use_cbam", False)
    ).to(device)

    mask_dir = cfg["data"].get("mask_dir", None)
    seg_model = None
    if mask_dir and str(mask_dir).lower() != "null":
        seg_model = DualInputSegmentationUNet(
            encoder_name=cfg["model"].get("encoder_name", "efficientnet-b0"),
            encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        ).to(device)

    # losses
    crit_cls = nn.BCEWithLogitsLoss()
    crit_seg = nn.BCEWithLogitsLoss() if seg_model is not None else None
    seg_w    = float(cfg["training"].get("seg_loss_weight", 1.0))

    # optimizer & scheduler
    params = list(clf.parameters()) + (list(seg_model.parameters()) if seg_model is not None else [])
    optimizer = torch.optim.AdamW(
        params, lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    scaler = GradScaler()
    best_score = -1.0  # we’ll track by F1 (or AUC if you prefer)

    num_epochs = int(cfg["training"]["num_epochs"])
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            clf, seg_model, train_loader, device,
            crit_cls, crit_seg, optimizer, scaler,
            seg_w=seg_w
        )

        metrics = validate(clf, seg_model, val_loader, device, crit_seg)
        acc, auc = metrics["accuracy"], metrics["auc"]
        precision, recall, f1 = metrics["precision"], metrics["recall"], metrics["f1"]
        mean_iou = metrics["mean_iou"]

        # choose primary validation score
        val_score = f1 if not np.isnan(f1) else (auc if not np.isnan(auc) else acc)

        print(f"TrainLoss {train_loss:.4f} | "
              f"Acc {acc*100:.2f}% | AUC {auc:.4f} | "
              f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f} | "
              f"IoU {mean_iou:.4f}" if mean_iou is not None else
              f"TrainLoss {train_loss:.4f} | "
              f"Acc {acc*100:.2f}% | AUC {auc:.4f} | "
              f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f}")

        # save "last" every epoch
        torch.save(clf.state_dict(), os.path.join(model_dir, "last_classifier.pth"))
        if seg_model is not None:
            torch.save(seg_model.state_dict(), os.path.join(model_dir, "last_segmenter.pth"))

        # save best by chosen metric
        if val_score > best_score:
            best_score = val_score
            torch.save(clf.state_dict(), os.path.join(model_dir, "best_classifier.pth"))
            if seg_model is not None:
                torch.save(seg_model.state_dict(), os.path.join(model_dir, "best_segmenter.pth"))
            print(f"✅ Saved new best (score={best_score:.4f})")

        scheduler.step(val_score)

    print("\nTraining complete.")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Checkpoints in: {model_dir}")


if __name__ == "__main__":
    main()
