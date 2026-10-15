from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a machine learning model",
    version="1.0.0",
)

MODEL_PATH = "house_price_model.joblib"
model = None

class HouseFeatures(BaseModel):
    area_sqft: float = Field(gt=0, description="Area in square feet")
    bedrooms: int = Field(ge=1, le=20)
    age_years: float = Field(ge=0)

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_price(house: HouseFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model first.",
        )

    features = [[house.area_sqft, house.bedrooms, house.age_years]]
    prediction = model.predict(features)[0]

    return {"predicted_price": round(float(prediction), 2)}
