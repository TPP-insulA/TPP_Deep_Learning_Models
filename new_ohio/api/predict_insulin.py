import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from stable_baselines3 import PPO
from pydantic import BaseModel, field_validator
from typing import List, Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your mobile app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = PPO.load("new_ohio/models/output/ppo_ohio.zip")

class InsulinCalculationRequest(BaseModel):
    date: str
    cgmPrev: List[float]
    glucoseObjective: float
    carbs: float
    insulinOnBoard: float
    sleepLevel: int
    workLevel: int
    activityLevel: int

    @field_validator('date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid date format. Please use ISO format (YYYY-MM-DDTHH:MM:SS)')

def prepare_observation(cgm_values, hour_of_day, carb_input, insulin_on_board):
    cgm_values = [float(x) for x in cgm_values]
    cgm_values = [np.log1p(x) for x in cgm_values]
    cgm_values = cgm_values + [0] * (24 - len(cgm_values)) if len(cgm_values) < 24 else cgm_values[:24]
    hour_of_day_norm = hour_of_day / 24.0
    carb_log = np.log1p(float(carb_input))
    iob_log = np.log1p(float(insulin_on_board))
    bolus_log = 0.0  # Para el modelo actual
    observation = np.concatenate([cgm_values, [hour_of_day_norm, bolus_log, carb_log, iob_log]])
    return observation.astype(np.float32)

@app.post("/predict_insulin")
async def predict_insulin_dose(data: InsulinCalculationRequest):
    try:
        # Get current hour from the request date
        request_date = datetime.datetime.fromisoformat(data.date.replace('Z', '+00:00'))
        hour_of_day = request_date.hour
        
        # Prepare observation for the model
        obs = prepare_observation(
            data.cgmPrev,
            hour_of_day,
            data.carbs,
            data.insulinOnBoard
        )
        
        # Get prediction from model
        action, _ = model.predict(obs, deterministic=True)
        total_dose = float(action[0])
        
        # Return the model's prediction directly
        return {
            "total": total_dose,
            "breakdown": {
                "correctionDose": total_dose,  # The model's prediction is the total dose
                "mealDose": 0.0,              # These are placeholders since the model
                "activityAdjustment": 0.0,     # doesn't provide a breakdown
                "timeAdjustment": 0.0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}