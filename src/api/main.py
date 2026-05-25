from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()


model = joblib.load("models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "NeuralRetail API is running"}

@app.post("/predict_churn")
def predict(recency: float, frequency: float, monetary: float):
    data = np.array([[recency, frequency, monetary]])
    pred = model.predict(data)[0]
    return {"churn": int(pred)}