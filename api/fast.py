import os

from sklearn import set_config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from claimlens.y_predict import load_model, predict_method

app = FastAPI()
set_config(transform_output="pandas")

MODEL_PATH = os.getenv("MODEL_PATH", "models/tfidf_balanced_ngrams.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Refund Intent API is running!"}


@app.get("/predict")
def predict_endpoint(
    customer_message: str = "input the text",
):
    loaded_model = load_model(MODEL_PATH)
    prediction = predict_method(loaded_model, customer_message)
    intent = int(prediction["class"])
    intent_probability = float(prediction["probability"])

    return {
        "Refund Intent": intent,
        'Refund Intent Probability': intent_probability
    }
