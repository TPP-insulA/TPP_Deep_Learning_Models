from api.classes import InputData
from api.model_loader import load_models
from custom.printer import print_error

from fastapi import APIRouter, FastAPI, HTTPException

app = FastAPI()
router = APIRouter()

@router.post("/predict")
async def predict(input_data: InputData):
    """
    Realiza predicciones utilizando los modelos cargados.
    
    Parámetros:
    -----------
    input_data : InputData
        Datos de entrada para el modelo.
        
    Retorna:
    --------
    dict
        Predicciones de los modelos.
    """
    try:
        # Cargar modelos
        models = load_models(input_data.framework, input_data.models)
        
        # Realizar predicciones
        predictions = {}
        for model in models:
            predictions[model] = model.predict(input_data.x_cgm, input_data.x_other)
        
        return {"predictions": predictions}
    
    except Exception as e:
        print_error(f"Error en la optimización de pesos: {e}")
        raise HTTPException(status_code=500, detail=str(e))