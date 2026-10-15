from data import CleanText
import joblib

call_model_path = "../../models/tfidf_balanced_ngrams.joblib"

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_method(model, client_input):
    clean_input = CleanText(client_input)
    predicted_class = model.predict(model, clean_input)
    predicted_probability = model.predict_proba(model, clean_input)
    return {"class":predicted_class, "probability":predicted_probability}
