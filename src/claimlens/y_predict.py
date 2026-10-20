from claimlens.data import CleanText
from claimlens.embeddings import embed_texts
import joblib
import pandas as pd
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "fully_trained_model_1.joblib"
THRESHOLD_PATH = ROOT / "models" / "fully_trained_model_1_threshold.joblib"

MODEL_PATH_2 = ROOT / "models" / "fully_trained_model_2.joblib"
THRESHOLD_PATH_2 = ROOT / "models" / "fully_trained_model_2_threshold.joblib"


cleaner = CleanText(remove_stop_words=False, lemmatize=False)


def load_model(model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH):
    """Load the trained model and its threshold."""
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    return model, threshold

def load_model_2(model_path=MODEL_PATH_2, threshold_path=THRESHOLD_PATH_2):
    """Load the trained model and its threshold."""
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    return model, threshold

def predict_method(model, threshold, client_input):

    embedding = embed_texts([cleaner.clean(client_input)])
    probability = model.predict_proba(embedding)[0][1]
    return {
        "refund_intent": int(probability >= threshold),
        "probability": float(probability),
    }

def predict_money_back(model, threshold, claim_text):
    embedding = embed_texts([cleaner.clean(claim_text)])
    probability = model.predict_proba(embedding)[0][1]
    return {
        "money_back": int(probability >= threshold),
        "probability": float(probability)
    }




if __name__ == "__main__":
    model, threshold = load_model()
    test_input = "I have a complaint. I need to speak to an agent. my package arrived wet and the leather jacket is destroyed"


    y_pred = predict_method(model, threshold, test_input)

    print(y_pred)
