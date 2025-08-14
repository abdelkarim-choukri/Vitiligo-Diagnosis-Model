import json, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, confusion_matrix, classification_report,
    average_precision_score, precision_recall_curve
)
from scripts.dataset import VitiligoDataset
from scripts.model_classification import DualEfficientNetClassifier

TARGET_SPEC = 0.90

p = yaml.safe_load(open("config/config.yaml"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

# dataset/loader
ds = VitiligoDataset(
    p["data"]["val_csv"],
    p["data"]["clinical_dir"],
    p["data"]["wood_dir"],
    mask_dir=None,
    transform=None,
    image_size=p["data"]["image_size"],
)
dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

# model
m = DualEfficientNetClassifier(use_cbam=p["model"].get("use_cbam", False)).to(dev).eval()
ckpt = p["paths"]["model_dir"] + "/best_classifier.pth"
try:
    state = torch.load(ckpt, map_location=dev, weights_only=True)
except TypeError:
    state = torch.load(ckpt, map_location=dev)
m.load_state_dict(state)

# collect scores
scores, labels = [], []
with torch.no_grad():
    for cli, woo, _, yy in dl:
        s = m(cli.to(dev), woo.to(dev)).float().cpu().numpy()
        scores.append(s); labels.append(yy.numpy())
scores = np.concatenate(scores)
labels = np.concatenate(labels)

# choose threshold
fpr, tpr, thr = roc_curve(labels, scores)
spec = 1 - fpr
ok = np.where(spec >= TARGET_SPEC)[0]
if len(ok):
    idx = ok[np.argmax(tpr[ok])]
    chosen_thr = float(thr[idx])
else:
    chosen_thr = 0.5  # fallback if target specificity is impossible on tiny val

# save threshold
with open("config/threshold.json", "w") as f:
    json.dump({"decision_threshold": chosen_thr, "target_specificity": TARGET_SPEC}, f, indent=2)

# report
yhat = (scores >= chosen_thr).astype(int)
cm = confusion_matrix(labels, yhat)
rep = classification_report(labels, yhat, digits=3)
ap  = average_precision_score(labels, scores)

print(f"Chosen threshold: {chosen_thr:.6f}")
print("Confusion matrix:\n", cm)
print(rep)
print("PR-AUC:", ap)
