from pydantic import BaseModel

class InputData(BaseModel):
    """
    Datos de entrada para el modelo.
    
    Atributos:
    ----------
    - framework : str
        Framework utilizado (ej. 'tensorflow', 'jax', 'pytorch').
    - models : list
        Lista de modelos a utilizar.
    - x_cgm : np.ndarray
        Datos CGM de forma (muestras, pasos_tiempo, características)
    - x_other : np.ndarray
        Otras características de forma (muestras, características)
    """
    framework: str
    models: list
    x_cgm: list
    x_other: list
    