import os
import argparse
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json

def load_data(data_dir):
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    # Load datasets
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Assign explicit labels
    df_fake["label"] = 0   # 0 = FAKE
    df_true["label"] = 1   # 1 = REAL

    # Standardize text column
    for df in [df_fake, df_true]:
        if "title" in df.columns and "text" in df.columns:
            df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "text" in df.columns:
            df["content"] = df["text"].fillna("")
        elif "content" in df.columns:
            df["content"] = df["content"].fillna("")
        else:
            raise ValueError("No valid text column found (need title/text/content).")

    # Merge datasets
    df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
    return df[["content", "label"]]

def train(data_dir, model_path, metrics_path="models/metrics.json"):
    # Load dataset
    df = load_data(data_dir)
    X = df["content"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], output_dict=True)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    # Save metrics
    metrics = {"accuracy": acc, "report": report}
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Model trained and saved to {model_path}")
    print(f"✅ Metrics saved to {metrics_path}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news detection model")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to data directory")
    parser.add_argument("--model_path", type=str, default="models/model.joblib", help="Path to save model")
    args = parser.parse_args()

    train(args.data_dir, args.model_path)
