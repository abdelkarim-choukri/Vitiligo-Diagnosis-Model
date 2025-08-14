import argparse, os, yaml, pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base config (YAML)")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base_csv = cfg["data"]["train_csv"]  # use your full (clean) CSV
    df = pd.read_csv(base_csv)
    assert "label" in df.columns, "CSV must contain a 'label' column"

    os.makedirs("config/folds", exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for i, (tr, va) in enumerate(skf.split(df, df["label"])):
        tr_csv = f"config/folds/train_fold{i}.csv"
        va_csv = f"config/folds/val_fold{i}.csv"
        df.iloc[tr].to_csv(tr_csv, index=False)
        df.iloc[va].to_csv(va_csv, index=False)

        fold_cfg = yaml.safe_load(open(args.config))
        fold_cfg["data"]["train_csv"] = tr_csv
        fold_cfg["data"]["val_csv"]   = va_csv
        fold_cfg["paths"]["model_dir"]   = f"/root/autodl-tmp/runs/fold{i}/models"
        fold_cfg["paths"]["results_dir"] = f"/root/autodl-tmp/runs/fold{i}/results"
        os.makedirs(fold_cfg["paths"]["model_dir"], exist_ok=True)
        os.makedirs(fold_cfg["paths"]["results_dir"], exist_ok=True)

        out_cfg = f"config/folds/config_fold{i}.yaml"
        with open(out_cfg, "w") as f:
            yaml.safe_dump(fold_cfg, f, sort_keys=False)

        print(f"[fold {i}] train={len(tr)} val={len(va)} -> {tr_csv}, {va_csv} | cfg={out_cfg}")

if __name__ == "__main__":
    main()
