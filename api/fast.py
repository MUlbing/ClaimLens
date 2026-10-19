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

app.state.model, app.state.threshold = load_model()


@app.get("/")
def root():
    return {"message": "Refund Intent API is running!"}


@app.get("/predict")
def predict_endpoint(
    customer_message: str = "input the text",
):
    result = predict_method(app.state.model, app.state.threshold, customer_message)

    intent = "Yes" if result["refund_intent"] == 1 else "No"
    return {
        "Refund Intent": intent,
        "Refund Intent Probability": round(result["probability"], 2),
    }
