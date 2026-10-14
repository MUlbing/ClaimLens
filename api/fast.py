from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    pickup_datetime: str = "input the text",
):
    intent = 'Yes'
    return {
        "Refund Intent": intent,
        'Refund Intent Probability': 0.95
    }
