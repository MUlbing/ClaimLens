from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from claimlens.y_predict import load_model, predict_method

app = FastAPI()

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
    loaded_model = load_model("/Users/marianelaulbing/code/MUlbing/ClaimLens/models/tfidf_balanced_ngrams.joblib")
    prediction = predict_method(loaded_model, customer_message)
    intent = prediction["class"]
    intent_probability = prediction["probability"]

    return {
        "Refund Intent": intent,
        'Refund Intent Probability': intent_probability
    }
