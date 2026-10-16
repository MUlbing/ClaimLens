from claimlens.data import CleanText
import joblib
import pandas as pd
from pathlib import Path

call_model_path = "/Users/marianelaulbing/code/MUlbing/ClaimLens/models/tfidf_balanced_ngrams.joblib"
cleaner = CleanText()

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_method(model, client_input):
    clean_input = cleaner.clean(client_input)
    print(clean_input)
    predicted_class = model.predict([clean_input])[0]
    predicted_probability = model.predict_proba([clean_input])[0][1]
    return {"class": predicted_class, "probability": predicted_probability}



if __name__ == "__main__":

    def test_function():
        test_input = "I have a complaint. I need to speak to an agent. my package arrived wet and the leather jacket is destroyed"

        test_model = load_model(call_model_path)

        y_pred = predict_method(test_model, test_input)

        print(y_pred)
    test_function()
