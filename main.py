from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained pipeline and MultiLabelBinarizer
model = joblib.load("Voting_model.pkl")
mlb = joblib.load("mlb.pkl")

app = FastAPI()

# ✅ Enable CORS for the specified origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://plant-recommender-dashboard.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class PollutionInput(BaseModel):
    PM2_5: float
    PM10: float
    NO: float
    NO2: float
    NOx: float
    NH3: float
    CO: float
    SO2: float
    O3: float
    Benzene: float
    Toluene: float
    Xylene: float
    AQI: float

@app.post("/predict")
def predict_pollution(data: PollutionInput):
    # Convert input data to numpy array with correct shape and column order
    input_array = np.array([[data.PM2_5, data.PM10, data.NO, data.NO2, data.NOx, data.NH3,
                             data.CO, data.SO2, data.O3, data.Benzene, data.Toluene,
                             data.Xylene, data.AQI]])
    
    # Get probabilities from the model
    probabilities = model.predict_proba(input_array)
    
    # For multi-label, use the maximum probability across all classifiers for each class
    avg_probs = np.max(probabilities, axis=1) if probabilities.ndim > 1 else probabilities
    
    # Ensure avg_probs is a 1D array for all classes
    avg_probs = avg_probs[0] if isinstance(avg_probs, np.ndarray) and avg_probs.ndim > 1 else avg_probs
    
    # Get top 3 plant recommendations
    top3_indices = np.argsort(avg_probs)[::-1][:3]
    recommendations = [
        {
            "plant": mlb.classes_[idx],
            "confidence": round(float(avg_probs[idx]), 2)
        }
        for idx in top3_indices
    ]

    # Get all plant predictions with confidence scores
    all_predictions = [
        {
            "plant": plant,
            "confidence": round(float(confidence), 2)
        }
        for plant, confidence in zip(mlb.classes_, avg_probs)
    ]

    return {
        "recommendations": recommendations,
        "all_predictions": all_predictions
    }

# Optional: Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}
