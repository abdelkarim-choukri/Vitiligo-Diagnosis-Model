A 3‑step quickstart:

pip install -r requirements.txt

Edit config/config.yaml if needed

Run in order:

python scripts/data_prep.py
python scripts/train.py --config config/config.yaml
python scripts/evaluate.py --config config/config.yaml

Note where to find logs, saved models (models/), and results (results/).