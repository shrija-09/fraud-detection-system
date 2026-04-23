from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: dict):
    try:
        values = list(transaction.values())
        features = np.array(values).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return {
            "fraud": int(prediction),
            "probability": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}