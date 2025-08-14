# eval_threshold.py
import yaml, torch, numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from scripts.dataset import VitiligoDataset
from scripts.model_classification import DualEfficientNetClassifier

p=yaml.safe_load(open("config/config.yaml"))
ds=VitiligoDataset(p["data"]["val_csv"], p["data"]["clinical_dir"], p["data"]["wood_dir"], None, None, p["data"]["image_size"])
dl=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0)
device="cuda" if torch.cuda.is_available() else "cpu"
m=DualEfficientNetClassifier(use_cbam=p["model"].get("use_cbam",False)).to(device).eval()
m.load_state_dict(torch.load(p["paths"]["model_dir"]+"/best_classifier.pth", map_location=device))
s,y=[],[]
with torch.no_grad():
    for cli,woo,_,yy in dl:
        s.append(m(cli.to(device), woo.to(device)).float().cpu().numpy())
        y.append(yy.numpy())
s=np.concatenate(s); y=np.concatenate(y)
fpr,tpr,thr = roc_curve(y, s)  # thresholds correspond to scores; lower thr = more positives
spec = 1 - fpr
ok = np.where(spec >= 0.90)[0]
th = thr[ok[np.argmax(tpr[ok])]] if len(ok) else 0.5
yhat = (s >= th).astype(int)
print(f"Chosen threshold={th:.4f}  specificity={spec[ok[np.argmax(tpr[ok])]] if len(ok) else spec.max():.3f}")
print(confusion_matrix(y, yhat))
print(classification_report(y, yhat, digits=3))
# also report PR-AUC (better for imbalance)
ap = average_precision_score(y, s)
pr, rc, th_pr = precision_recall_curve(y, s)
print("PR-AUC:", ap)
