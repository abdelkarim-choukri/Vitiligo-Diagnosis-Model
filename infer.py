import sys, yaml, json, torch
from PIL import Image
import torchvision.transforms as T
from scripts.model_classification import DualEfficientNetClassifier

def load_rgb(p):
    img = Image.open(p); img.load(); return img.convert('RGB')

if __name__ == "__main__":
    clinical, wood = sys.argv[1], sys.argv[2]
    p = yaml.safe_load(open("config/config.yaml"))
    thr = json.load(open("config/threshold.json"))["decision_threshold"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tf = T.Compose([
        T.Resize((int(p["data"]["image_size"]), int(p["data"]["image_size"]))),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    c = tf(load_rgb(clinical)).unsqueeze(0).to(dev)
    w = tf(load_rgb(wood)).unsqueeze(0).to(dev)

    m = DualEfficientNetClassifier(use_cbam=p["model"].get("use_cbam", False)).to(dev).eval()
    m.load_state_dict(torch.load(p["paths"]["model_dir"]+"/best_classifier.pth", map_location=dev))
    with torch.no_grad():
        s = float(m(c, w).cpu())
    y = int(s >= thr)
    print(f"score={s:.6f}  threshold={thr:.6f}  pred={y} (1=positive)")
