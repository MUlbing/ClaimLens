from claimlens.data import CleanText
import joblib
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "tfidf_balanced_ngrams.joblib"
THRESHOLD_PATH = ROOT / "models" / "tfidf_balanced_ngrams_threshold.joblib"

cleaner = CleanText()

def load_model(model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH):
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    return model, threshold

def predict_method(model, threshold, client_input):
    clean_input = cleaner.clean(client_input)
    print(clean_input)
    # predicted_class = model.predict([clean_input])[0]
    probability = model.predict_proba([clean_input])[0][1]
    return {
        "Refund Intent": int(probability >= threshold), "probability": probability}



if __name__ == "__main__":
    model, threshold = load_model()
    test_input = "I have a complaint. I need to speak to an agent. my package arrived wet and the leather jacket is destroyed"


    y_pred = predict_method(model, threshold, test_input)

    print(y_pred)
