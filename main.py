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
    print("Raw Probabilities:", probabilities)  # Debug output
    
    # Revert to original logic
    avg_probs = np.array([p[0] if isinstance(p, list) else p for p in probabilities]).flatten()
    print("Averaged Probabilities:", avg_probs)  # Debug output
    
    # Get top 3 plant recommendations
    top3_indices = avg_probs.argsort()[::-1][:3]
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