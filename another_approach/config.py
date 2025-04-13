import os

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Global configuration
CONFIG = {
    "data_path": os.path.join(PROJECT_DIR, "data", "subjects"),
    "processed_data_path": os.path.join(PROJECT_DIR, "data", "processed"),
    "params_path": os.path.join(PROJECT_DIR, "data", "params")
}


WINDOW_PREV_HOURS = 2  # Ventana previa de 2 horas (parametrizable)
WINDOW_POST_HOURS = 2  # Ventana posterior de 2 horas (parametrizable)
IOB_WINDOW_HOURS = 4   # Ventana de 4 horas para insulinOnBoard
SAMPLES_PER_HOUR = 12  # 1 dato cada 5 min = 12 datos por hora
PREV_SAMPLES = WINDOW_PREV_HOURS * SAMPLES_PER_HOUR  # 24 datos previos
POST_SAMPLES = WINDOW_POST_HOURS * SAMPLES_PER_HOUR  # 24 datos posteriores