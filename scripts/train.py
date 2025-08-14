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
            
            logits = clf(cli, woo)  # [B]         # [B]
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
        
        logits = clf(cli, woo)
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


# def main():
#     import argparse, os, numpy as np, torch
#     import torch.nn as nn
#     from torch.utils.data import DataLoader
#     from torch.cuda.amp import GradScaler  # keep legacy API to match your train/validate

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config/config.yaml")
#     args = parser.parse_args()

#     cfg = load_config(args.config)

#     # device & speed knobs
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.backends.cudnn.benchmark = True
#     try:
#         torch.set_float32_matmul_precision("high")  # safe no-op on CPU/older GPUs
#     except Exception:
#         pass

#     # paths
#     model_dir   = cfg["paths"]["model_dir"]
#     results_dir = cfg["paths"]["results_dir"]
#     ensure_dir(model_dir); ensure_dir(results_dir)

#     # -----------------------------------
#     # DATA
#     # -----------------------------------
#     # Paired augmentations for train (same geom/color on clinical & wood)
#     try:
#         import cv2
#         import albumentations as A
#         image_size = int(cfg["data"]["image_size"])
#         tfm = A.Compose([
#             A.RandomResizedCrop(size=(image_size, image_size),
#                                 scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.7),
#             A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=10,
#                                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
#             A.HorizontalFlip(p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
#         ], additional_targets={'image2': 'image'})
#     except Exception as e:
#         print(f"[warn] Albumentations not available/compatible ({e}); continuing without train-time augs.")
#         tfm = None
#         image_size = int(cfg["data"]["image_size"])

#     train_ds = VitiligoDataset(
#         csv_file=cfg["data"]["train_csv"],
#         clinical_dir=cfg["data"]["clinical_dir"],
#         wood_dir=cfg["data"]["wood_dir"],
#         mask_dir=cfg["data"].get("mask_dir", None),
#         transform=tfm,                          # augs ON for train
#         image_size=image_size,
#     )
#     val_ds = VitiligoDataset(
#         csv_file=cfg["data"]["val_csv"],
#         clinical_dir=cfg["data"]["clinical_dir"],
#         wood_dir=cfg["data"]["wood_dir"],
#         mask_dir=cfg["data"].get("mask_dir", None),
#         transform=None,                         # augs OFF for val
#         image_size=image_size,
#     )

#     # DataLoaders
#     num_workers = int(os.environ.get("NUM_WORKERS", "8"))
#     bs          = int(cfg["training"]["batch_size"])
#     persist     = (num_workers > 0)  # must be False when num_workers==0

#     train_loader = DataLoader(
#         train_ds, batch_size=bs, shuffle=True,
#         num_workers=num_workers, pin_memory=True, persistent_workers=persist
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=bs, shuffle=False,
#         num_workers=num_workers, pin_memory=True, persistent_workers=persist
#     )

#     # -----------------------------------
#     # MODELS
#     # -----------------------------------
#     clf = DualEfficientNetClassifier(
#         use_cbam=cfg["model"].get("use_cbam", False)
#     ).to(device)

#     seg_model = None
#     mask_dir  = cfg["data"].get("mask_dir", None)
#     if mask_dir and str(mask_dir).lower() != "null":
#         seg_model = DualInputSegmentationUNet(
#             encoder_name=cfg["model"].get("encoder_name", "efficientnet-b0"),
#             encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
#         ).to(device)

#     # -----------------------------------
#     # LOSSES (handle class imbalance)
#     # -----------------------------------
#     try:
#         import pandas as _pd
#         _df  = _pd.read_csv(cfg["data"]["train_csv"])
#         _neg = int((_df["label"] == 0).sum())
#         _pos = int((_df["label"] == 1).sum())
#         _pw  = float(_neg / max(_pos, 1)) if _pos > 0 else 1.0
#         print(f"[info] pos_weight={_pw:.3f}  (neg={_neg}, pos={_pos})")
#         pos_weight = torch.tensor([_pw], device=device)
#     except Exception as e:
#         print(f"[warn] could not compute pos_weight ({e}); using 1.0")
#         pos_weight = torch.tensor([1.0], device=device)

#     crit_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     crit_seg = nn.BCEWithLogitsLoss() if seg_model is not None else None
#     seg_w    = float(cfg["training"].get("seg_loss_weight", 0.0))

#     # -----------------------------------
#     # OPTIM / SCHED / AMP
#     # -----------------------------------
#     lr = float(cfg["training"]["learning_rate"])
#     wd = float(cfg["training"].get("weight_decay", 1e-4))
#     params = list(clf.parameters()) + (list(seg_model.parameters()) if seg_model is not None else [])
#     optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="max", factor=0.5, patience=2, verbose=True
#     )

#     scaler = GradScaler(enabled=(device.type == "cuda"))

#     # -----------------------------------
#     # TRAIN LOOP
#     # -----------------------------------
#     best_score = -1.0
#     num_epochs = int(cfg["training"]["num_epochs"])

#     for epoch in range(1, num_epochs + 1):
#         print(f"\nEpoch {epoch}/{num_epochs}")

#         train_loss = train_one_epoch(
#             clf, seg_model, train_loader, device,
#             crit_cls, crit_seg, optimizer, scaler,
#             seg_w=seg_w
#         )

#         metrics = validate(clf, seg_model, val_loader, device, crit_seg)
#         acc, auc = metrics.get("accuracy"), metrics.get("auc")
#         precision, recall, f1 = metrics.get("precision"), metrics.get("recall"), metrics.get("f1")
#         mean_iou = metrics.get("mean_iou")

#         # primary validation score:
#         # prefer F1 only if it's > 0; otherwise use AUC, else accuracy
#         if (f1 is not None) and (not np.isnan(f1)) and (f1 > 0.0):
#             val_score = f1
#         elif (auc is not None) and (not np.isnan(auc)):
#             val_score = auc
#         else:
#             val_score = acc if acc is not None else 0.0

#         if mean_iou is not None:
#             print(f"TrainLoss {train_loss:.4f} | Acc {acc*100:.2f}% | AUC {auc:.4f} | "
#                   f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f} | IoU {mean_iou:.4f}")
#         else:
#             print(f"TrainLoss {train_loss:.4f} | Acc {acc*100:.2f}% | AUC {auc:.4f} | "
#                   f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f}")

#         # save "last" every epoch
#         torch.save(clf.state_dict(), os.path.join(model_dir, "last_classifier.pth"))
#         if seg_model is not None:
#             torch.save(seg_model.state_dict(), os.path.join(model_dir, "last_segmenter.pth"))

#         # save best by chosen metric
#         if val_score > best_score:
#             best_score = val_score
#             torch.save(clf.state_dict(), os.path.join(model_dir, "best_classifier.pth"))
#             if seg_model is not None:
#                 torch.save(seg_model.state_dict(), os.path.join(model_dir, "best_segmenter.pth"))
#             print(f"✅ Saved new best (score={best_score:.4f})")

#         # step ReduceLROnPlateau with the validation metric
#         scheduler.step(val_score)

#     print("\nTraining complete.")
#     print(f"Best validation score: {best_score:.4f}")
#     print(f"Checkpoints in: {model_dir}")


def main():
    import argparse, os, numpy as np, torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler  # keep this to match your train/validate code

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # -----------------------------
    # Device & speed knobs
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # safe no-op if unsupported
    except Exception:
        pass

    # If CUDA exists but kernels are incompatible (e.g., RTX 5090 on old wheels), fall back to CPU.
    if device.type == "cuda":
        try:
            x = torch.randn(1, 3, 8, 8, device="cuda")
            torch.nn.Conv2d(3, 3, 3, padding=1).to("cuda")(x)
        except Exception as e:
            print(f"[warn] CUDA present but unusable ({e}); falling back to CPU.")
            device = torch.device("cpu")

    # -----------------------------
    # Paths
    # -----------------------------
    model_dir   = cfg["paths"]["model_dir"]
    results_dir = cfg["paths"]["results_dir"]
    ensure_dir(model_dir); ensure_dir(results_dir)

    # -----------------------------
    # DATA (paired augs for train)
    # -----------------------------
    try:
        import cv2, albumentations as A
        image_size = int(cfg["data"]["image_size"])
        tfm = A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size),
                                scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.7),
            # v2-safe usage (warning-only to prefer Affine; still works)
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=10,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        ], additional_targets={'image2': 'image'})
    except Exception as e:
        print(f"[warn] Albumentations not available/compatible ({e}); continuing without train-time augs.")
        tfm = None
        image_size = int(cfg["data"]["image_size"])

    train_ds = VitiligoDataset(
        csv_file=cfg["data"]["train_csv"],
        clinical_dir=cfg["data"]["clinical_dir"],
        wood_dir=cfg["data"]["wood_dir"],
        mask_dir=cfg["data"].get("mask_dir", None),
        transform=tfm,                          # augs ON for train
        image_size=image_size,
    )
    val_ds = VitiligoDataset(
        csv_file=cfg["data"]["val_csv"],
        clinical_dir=cfg["data"]["clinical_dir"],
        wood_dir=cfg["data"]["wood_dir"],
        mask_dir=cfg["data"].get("mask_dir", None),
        transform=None,                         # augs OFF for val
        image_size=image_size,
    )

    num_workers = int(os.environ.get("NUM_WORKERS", "8"))
    bs          = int(cfg["training"]["batch_size"])
    persist     = (num_workers > 0)  # must be False when num_workers==0

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persist
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persist
    )

    # -----------------------------
    # MODELS
    # -----------------------------
    clf = DualEfficientNetClassifier(
        use_cbam=cfg["model"].get("use_cbam", False)
    ).to(device)

    seg_model = None
    mask_dir  = cfg["data"].get("mask_dir", None)
    if mask_dir and str(mask_dir).lower() != "null":
        seg_model = DualInputSegmentationUNet(
            encoder_name=cfg["model"].get("encoder_name", "efficientnet-b0"),
            encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        ).to(device)

    # -----------------------------
    # LOSSES (class imbalance)
    # -----------------------------
    try:
        import pandas as _pd
        _df  = _pd.read_csv(cfg["data"]["train_csv"])
        _neg = int((_df["label"] == 0).sum())
        _pos = int((_df["label"] == 1).sum())
        _pw  = float(_neg / max(_pos, 1)) if _pos > 0 else 1.0
        print(f"[info] pos_weight={_pw:.3f}  (neg={_neg}, pos={_pos})")
        pos_weight = torch.tensor([_pw], device=device)
    except Exception as e:
        print(f"[warn] could not compute pos_weight ({e}); using 1.0")
        pos_weight = torch.tensor([1.0], device=device)

    crit_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_seg = nn.BCEWithLogitsLoss() if seg_model is not None else None
    seg_w    = float(cfg["training"].get("seg_loss_weight", 0.0))

    # -----------------------------
    # OPTIM / SCHED / AMP
    # -----------------------------
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"].get("weight_decay", 1e-4))
    params = list(clf.parameters()) + (list(seg_model.parameters()) if seg_model is not None else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # Your torch version doesn't accept 'verbose', so omit it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    def _get_lr(opt):
        return float(opt.param_groups[0]["lr"])
    last_lr = _get_lr(optimizer)

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    best_score = -1.0
    num_epochs = int(cfg["training"]["num_epochs"])

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            clf, seg_model, train_loader, device,
            crit_cls, crit_seg, optimizer, scaler,
            seg_w=seg_w
        )

        metrics = validate(clf, seg_model, val_loader, device, crit_seg)
        acc, auc = metrics.get("accuracy"), metrics.get("auc")
        precision, recall, f1 = metrics.get("precision"), metrics.get("recall"), metrics.get("f1")
        mean_iou = metrics.get("mean_iou")

        # Prefer F1 only if it's > 0; otherwise use AUC; else accuracy
        if (f1 is not None) and (not np.isnan(f1)) and (f1 > 0.0):
            val_score = f1
        elif (auc is not None) and (not np.isnan(auc)):
            val_score = auc
        else:
            val_score = acc if acc is not None else 0.0

        if mean_iou is not None:
            print(f"TrainLoss {train_loss:.4f} | Acc {acc*100:.2f}% | AUC {auc:.4f} | "
                  f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f} | IoU {mean_iou:.4f}")
        else:
            print(f"TrainLoss {train_loss:.4f} | Acc {acc*100:.2f}% | AUC {auc:.4f} | "
                  f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f}")

        # Save "last" every epoch
        torch.save(clf.state_dict(), os.path.join(model_dir, "last_classifier.pth"))
        if seg_model is not None:
            torch.save(seg_model.state_dict(), os.path.join(model_dir, "last_segmenter.pth"))

        # Save "best" by chosen metric
        if val_score > best_score:
            best_score = val_score
            torch.save(clf.state_dict(), os.path.join(model_dir, "best_classifier.pth"))
            if seg_model is not None:
                torch.save(seg_model.state_dict(), os.path.join(model_dir, "best_segmenter.pth"))
            print(f"✅ Saved new best (score={best_score:.4f})")

        # Step LR scheduler and log LR drops
        scheduler.step(val_score)
        new_lr = _get_lr(optimizer)
        if new_lr < last_lr:
            print(f"🔻 LR reduced: {last_lr:.6g} → {new_lr:.6g}")
        last_lr = new_lr

    print("\nTraining complete.")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Checkpoints in: {model_dir}")

if __name__ == "__main__":
    main()
