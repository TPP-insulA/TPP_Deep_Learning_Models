import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from stable_baselines3 import PPO
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union
import logging
from new_ohio.models.ppo import predict_insulin_dose as model_predict_insulin_dose, prepare_enhanced_observation

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your mobile app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar logging para la API
api_logger = logging.getLogger("api_logger")
api_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
api_logger.addHandler(handler)

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
    user_profile: Optional[str] = None

    @field_validator('date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid date format. Please use ISO format (YYYY-MM-DDTHH:MM:SS)')

def apply_elderly_rules(current_cgm: float) -> float:
    """Aplica la tabla de reglas de dosificación para la paciente anciana."""
    if current_cgm < 150:
        return 0.0
    elif current_cgm <= 200:
        return 2.0
    elif current_cgm <= 250:
        return 4.0
    elif current_cgm <= 300:
        return 6.0
    else: # más de 300 mg/dL
        return 8.0

@app.post("/predict_insulin")
async def predict_insulin(data: InsulinCalculationRequest):
    try:
        request_date = datetime.datetime.fromisoformat(data.date.replace('Z', '+00:00'))
        current_cgm = float(data.cgmPrev[-1]) if data.cgmPrev else 120.0
        api_logger.info(f"Petición recibida: CGM={current_cgm}, Carbs={data.carbs}, IOB={data.insulinOnBoard}, Perfil={data.user_profile}")

        total_dose: float

        if data.user_profile == "elderly_rules":
            api_logger.info(f"Aplicando reglas para perfil 'elderly_rules'. CGM actual: {current_cgm}")
            total_dose = apply_elderly_rules(current_cgm)
            api_logger.info(f"Dosis por reglas 'elderly_rules': {total_dose}")
        else:
            api_logger.info("Usando modelo PPO para predicción.")
            obs = prepare_enhanced_observation(
                cgm_values=data.cgmPrev,
                carb_input=data.carbs,
                iob=data.insulinOnBoard,
                timestamp=request_date,
                meal_carbs=data.carbs,
                meal_time_diff=0,
                has_meal=1.0 if data.carbs > 0 else 0.0,
                meals_in_window=1 if data.carbs > 0 else 0,
                extended_cgm=data.cgmPrev
            )
            
            total_dose = model_predict_insulin_dose(
                model_path="new_ohio/models/enhanced_output/final_model.zip",
                observation=obs,
                current_cgm=current_cgm,
                carbs=data.carbs,
                iob=data.insulinOnBoard
            )
            api_logger.info(f"Dosis predicha por modelo PPO (post-validación): {total_dose}")

        return {
            "total": total_dose,
            "breakdown": {
                "correctionDose": total_dose,
                "mealDose": 0.0,
                "activityAdjustment": 0.0,
                "timeAdjustment": 0.0
            },
            "profile_applied": data.user_profile if data.user_profile == "elderly_rules" else "default_ml"
        }
    except Exception as e:
        api_logger.error(f"Error en /predict_insulin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}