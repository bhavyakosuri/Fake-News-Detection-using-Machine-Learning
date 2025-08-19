# Fake-News-Detection-using-MachineLearning(Passive Aggressive Classifier)— Flask App

End-to-end fake news detector trained on the Kaggle dataset
**[`emineyetm/fake-news-detection-datasets`](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)**,
featuring preprocessing, TF‑IDF feature extraction, a Passive Aggressive Classifier, evaluation,
and a Flask web UI for real-time predictions.

## Project Structure
```
fake-news-ml-flask/
├── app.py                 # Flask app for real-time predictions
├── train.py               # Train/evaluate and export model pipeline
├── inference.py           # CLI predictions from a text file or stdin
├── requirements.txt
├── Dockerfile
├── gunicorn.conf.py
├── Procfile
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── models/
│   └── model.joblib       # created after training
├── data/
│   ├── raw/               # put Kaggle CSVs here (or auto-download via Kaggle API)
│   └── processed/
└── tests/
    └── test_smoke.py
```

## 1) Getting the data

Option A — **Kaggle API (recommended):**
1. Install the CLI and authenticate: place your `kaggle.json` in `~/.kaggle/`.
2. From the project root, run:
   ```bash
   kaggle datasets download -d emineyetm/fake-news-detection-datasets -p data/raw -unzip
   ```
   You should see files like `Fake.csv` and `True.csv` in `data/raw/`.

Option B — **Manual download:**
- Download from the Kaggle link and extract all CSVs into `data/raw/`.

> The training script is robust: it will autodetect labels from a `label` column,
> or infer labels from filenames like `Fake.csv` (label=0/“FAKE”) and `True.csv` (label=1/“REAL”).

## 2) Train the model
```bash
python -m pip install -r requirements.txt
python train.py --data_dir data/raw --model_path models/model.joblib
```
This will:
- Load + clean the data
- Split into train/validation
- Build a **TF‑IDF (1–2 grams)** + **PassiveAggressiveClassifier** pipeline
- Evaluate with accuracy, precision, recall, F1, and a confusion matrix
- Save `models/model.joblib` and `metrics.json`

## 3) Run the Flask app (development)
```bash
python app.py
```
Open http://127.0.0.1:5000 and paste a news article to get **REAL/FAKE** predictions.

## 4) Production (Gunicorn + Docker)

### Local Gunicorn
```bash
python -m pip install -r requirements.txt
gunicorn -c gunicorn.conf.py app:app
```

### Docker
```bash
docker build -t fake-news-ml-flask .
docker run -p 8000:8000 fake-news-ml-flask
```
Open http://127.0.0.1:8000/

## 5) CLI predictions
```bash
echo "Your news text here..." | python inference.py
# or
python inference.py --file example.txt
```

## Notes
- The model uses `stop_words='english'`, sublinear TF, and bi-grams for stronger performance.
- `PassiveAggressiveClassifier` is strong for linear text classification and handles large sparse features well.
- For reproducibility, default `random_state=42` is used and stratified splits are applied.

