import argparse, glob, os, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="config/folds/config_fold*.yaml",
                    help="glob for fold configs")
    ap.add_argument("--cpu_only", action="store_true",
                    help="set CUDA_VISIBLE_DEVICES='' to force CPU")
    args = ap.parse_args()

    cfgs = sorted(glob.glob(args.pattern))
    if not cfgs:
        print("No fold configs found. Run scripts/make_folds.py first.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    if args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""

    for cf in cfgs:
        print(f"\n=== Training {cf} ===")
        ret = subprocess.call([sys.executable, "-m", "scripts.train", "--config", cf], env=env)
        if ret != 0:
            print(f"Training failed for {cf} (exit {ret})", file=sys.stderr)
            sys.exit(ret)
    print("\nAll folds finished.")

if __name__ == "__main__":
    main()
