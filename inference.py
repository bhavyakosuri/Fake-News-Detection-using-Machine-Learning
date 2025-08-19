import argparse
import sys
from joblib import load

def read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/model.joblib")
    parser.add_argument("--file", type=str, default=None, help="Text file with news content. If omitted, read from stdin.")
    args = parser.parse_args()

    model = load(args.model_path)
    if args.file:
        text = read_text_from_file(args.file)
    else:
        text = sys.stdin.read()

    pred = model.predict([text])[0]
    label = "REAL" if int(pred) == 1 else "FAKE"
    print(label)

if __name__ == "__main__":
    main()
