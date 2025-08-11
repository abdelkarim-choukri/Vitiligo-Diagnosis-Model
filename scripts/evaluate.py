# scripts/evaluate.py

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.dataset import VitiligoDataset
from scripts.model_classification import DualEfficientNetClassifier
from scripts.model_segmentation import DualInputSegmentationUNet

def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths from config
    test_csv     = cfg['data']['test_csv']
    clinical_dir = cfg['data']['clinical_dir']
    wood_dir     = cfg['data']['wood_dir']
    mask_dir     = cfg['data'].get('mask_dir', None)

    # Dataset & loader
    test_ds = VitiligoDataset(
        csv_file=test_csv,
        clinical_dir=clinical_dir,
        wood_dir=wood_dir,
        mask_dir=mask_dir,
        transform=None,               # no augmentations for eval
        image_size=cfg['data']['image_size']
    )
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'],
                             shuffle=False, num_workers=4)

    # Load classifier
    clf = DualEfficientNetClassifier(
        use_cbam=cfg['model']['use_cbam'],
        pretrained=False
    ).to(device)
    clf_ckpt = os.path.join("models", "best_classifier.pth")
    clf.load_state_dict(torch.load(clf_ckpt, map_location=device))
    clf.eval()

    # Load segmenter if masks exist
    seg = None
    if mask_dir:
        seg = DualInputSegmentationUNet().to(device)
        seg_ckpt = os.path.join("models", "best_segmenter.pth")
        seg.load_state_dict(torch.load(seg_ckpt, map_location=device))
        seg.eval()

    all_labels,  all_preds,  all_probs = [], [], []
    all_ious = []

    with torch.no_grad():
        for cli, woo, mask, label in test_loader:
            cli, woo = cli.to(device), woo.to(device)
            labels_np = label.numpy()
            # Classification
            logits = clf(cli, woo)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)

            all_labels.extend(labels_np.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            # Segmentation IoU
            if seg and mask_dir:
                mask = mask.to(device).float()
                mask_pred = (torch.sigmoid(seg(cli, woo)) >= 0.5).cpu().numpy().astype(int)
                mask_true = mask.cpu().numpy().astype(int)
                # Compute per-sample IoU
                for mp, mt in zip(mask_pred, mask_true):
                    inter = (mp & mt).sum()
                    union = (mp | mt).sum()
                    if union == 0:
                        iou = 1.0 if inter == 0 else 0.0
                    else:
                        iou = inter / union
                    all_ious.append(iou)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    mean_iou = np.mean(all_ious) if all_ious else None

    print("=== Test Set Evaluation ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"AUC:      {auc:.4f}")
    if mean_iou is not None:
        print(f"Mean IoU: {mean_iou:.4f}")

    # Save detailed results
    results_df = pd.DataFrame({
        'clinical':  test_ds.df['clinical_filename'],
        'wood':      test_ds.df['wood_filename'],
        'label':     all_labels,
        'pred':      all_preds,
        'prob':      all_probs
    })
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/test_results.csv", index=False)
    print("Detailed per-sample results saved to results/test_results.csv")

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(all_labels, all_preds)
    recall    = recall_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

if __name__ == "__main__":
    main()
