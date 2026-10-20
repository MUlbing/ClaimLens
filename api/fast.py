import os

from sklearn import set_config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from claimlens.y_predict import load_model, predict_method, load_model_2, predict_money_back


app = FastAPI()
set_config(transform_output="pandas")

MODEL_PATH_1 = os.getenv("MODEL_PATH", "models/fully_trained_model_1.joblib")
MODEL_PATH_2 = os.getenv("MODEL_PATH", "models/fully_trained_model_2.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model, app.state.threshold = load_model()
app.state.model_2, app.state.threshold_2 = load_model_2()


@app.get("/")
def root():
    return {"message": "Refund Intent API is running!"}


@app.get("/predict")
def predict_endpoint(
    customer_message: str = "Explain in one sentence your problem"
):
    result = predict_method(app.state.model, app.state.threshold, customer_message)

    intent = "Yes" if result["refund_intent"] == 1 else "No"
    return {
        "Refund Intent": intent,
        "Refund Intent Probability": round(result["probability"], 2),
    }

@app.get("/more_context")
def concat_user_input(
    customer_message: str = "Give some more context about the issue at hand."
):
    result = predict_money_back(app.state.model_2, app.state.threshold_2, customer_message)
    intent = "Yes" if result["money_back"] == 1 else "No"
    return {
        "Money Back": intent,
        "Chance of money loss": round(result["probability"], 2),
    }
