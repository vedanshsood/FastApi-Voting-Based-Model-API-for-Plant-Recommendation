from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained pipeline and MultiLabelBinarizer
model = joblib.load("Voting_model.pkl")
mlb = joblib.load("mlb.pkl")

app = FastAPI()

# ✅ Enable CORS for all origins (change "*" to your frontend URL for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
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
    input_array = np.array([[data.PM2_5, data.PM10, data.NO, data.NO2, data.NOx, data.NH3,
                             data.CO, data.SO2, data.O3, data.Benzene, data.Toluene,
                             data.Xylene, data.AQI]])
    
    probabilities = model.predict_proba(input_array)
    avg_probs = np.array([p[0] if isinstance(p, list) else p for p in probabilities]).flatten()
    top3_indices = avg_probs.argsort()[::-1][:3]
    
    recommendations = []
    for idx in top3_indices:
        recommendations.append({
            "plant": mlb.classes_[idx],
            "confidence": round(avg_probs[idx], 2)
        })

    return {"recommendations": recommendations}
