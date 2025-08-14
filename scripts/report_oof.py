import argparse, glob, json, os, yaml, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from scripts.dataset import VitiligoDataset
from scripts.model_classification import DualEfficientNetClassifier

def specificity(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def choose_threshold(y, s, target_spec=0.90):
    """Pick threshold with spec>=target having max TPR; else fall back to Youden J."""
    fpr, tpr, thr = roc_curve(y, s)  # thr[0]=inf
    spec = 1 - fpr
    ok = np.where(spec >= target_spec)[0]
    if ok.size:
        idx = ok[np.argmax(tpr[ok])]
        return float(thr[idx]), {"tpr": float(tpr[idx]), "spec": float(spec[idx])}
    # fallback: Youden J
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), {"tpr": float(tpr[idx]), "spec": float(spec[idx])}

def eval_fold(cf_path, device):
    p = yaml.safe_load(open(cf_path))
    ds = VitiligoDataset(
        p["data"]["val_csv"], p["data"]["clinical_dir"], p["data"]["wood_dir"],
        mask_dir=None, transform=None, image_size=p["data"]["image_size"]
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    m = DualEfficientNetClassifier(use_cbam=p["model"].get("use_cbam", False)).to(device).eval()
    ckpt = os.path.join(p["paths"]["model_dir"], "best_classifier.pth")
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state)

    scores, labels = [], []
    with torch.no_grad():
        for cli, woo, _, y in dl:
            scores.append(m(cli.to(device), woo.to(device)).float().cpu().numpy())
            labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)

    # Also capture filenames (same order as CSV)
    df_val = pd.read_csv(p["data"]["val_csv"])
    return s, y, df_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="config/folds/config_fold*.yaml")
    ap.add_argument("--target_spec", type=float, default=0.90)
    ap.add_argument("--out_json", default="config/threshold_oof.json")
    ap.add_argument("--out_csv",  default="config/oof_predictions.csv")
    ap.add_argument("--out_txt",  default="config/oof_report.txt")
    ap.add_argument("--cpu_only", action="store_true")
    args = ap.parse_args()

    cfgs = sorted(glob.glob(args.pattern))
    assert cfgs, "No fold configs found. Run scripts/make_folds.py and train folds."

    device = "cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda"

    all_s, all_y, parts = [], [], []
    for i, cf in enumerate(cfgs):
        s, y, df_val = eval_fold(cf, device)
        all_s.append(s); all_y.append(y)
        df = df_val.copy()
        df["fold"] = i
        df["score"] = s
        parts.append(df)

    S = np.concatenate(all_s)
    Y = np.concatenate(all_y)
    df_all = pd.concat(parts, ignore_index=True)

    # AUCs on OOF
    roc_auc = float(roc_auc_score(Y, S)) if len(np.unique(Y)) > 1 else float("nan")
    pr_auc  = float(average_precision_score(Y, S))

    thr, stats = choose_threshold(Y, S, target_spec=args.target_spec)
    yhat = (S >= thr).astype(int)
    cm = confusion_matrix(Y, yhat)
    rep = classification_report(Y, yhat, digits=3)

    # Save threshold + summary JSON
    out = {
        "decision_threshold": thr,
        "target_specificity": args.target_spec,
        "achieved_specificity": specificity(cm),
        "achieved_sensitivity": (cm[1,1] / (cm[1,1] + cm[1,0])) if (cm[1,1] + cm[1,0])>0 else 0.0,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tpr_at_choice": stats.get("tpr"),
        "spec_at_choice": stats.get("spec"),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(Y)),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Save OOF predictions CSV (useful for error analysis)
    df_all["pred"] = (df_all["score"] >= thr).astype(int)
    df_all.to_csv(args.out_csv, index=False)

    # Save a human-readable TXT report
    with open(args.out_txt, "w") as f:
        f.write(f"OOF ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"OOF PR-AUC : {pr_auc:.4f}\n")
        f.write(f"Chosen threshold: {thr:.6f}\n")
        f.write(f"Confusion matrix (OOF):\n{cm}\n\n")
        f.write(rep + "\n")

    print(json.dumps(out, indent=2))
    print("\nClassification report (OOF):\n" + rep)
    print(f"\nSaved:\n  {args.out_json}\n  {args.out_csv}\n  {args.out_txt}")

if __name__ == "__main__":
    main()
