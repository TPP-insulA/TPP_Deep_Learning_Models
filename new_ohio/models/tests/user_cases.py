# Casos de usuario simulando la app: solo glucosas recientes y carbs
USER_CASES = [
    {
        'name': 'Glucosa estable, comida moderada',
        'glucoses': [120.0]*12,
        'carbs': 40.0
    },
    {
        'name': 'Glucosa alta sostenida, comida grande',
        'glucoses': [200.0]*12,
        'carbs': 80.0
    },
    {
        'name': 'Glucosa descendente, comida pequeña',
        'glucoses': [150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 95.0, 90.0, 85.0, 80.0, 78.0, 75.0],
        'carbs': 15.0
    },
    {
        'name': 'Glucosa baja, sin comida',
        'glucoses': [70.0]*12,
        'carbs': 0.0
    },
    {
        'name': 'Glucosa variable, comida normal',
        'glucoses': [110.0, 115.0, 120.0, 130.0, 125.0, 140.0, 135.0, 120.0, 110.0, 100.0, 105.0, 115.0],
        'carbs': 50.0
    },
    {
        'name': 'Glucosa muy alta, comida pequeña',
        'glucoses': [300.0]*12,
        'carbs': 10.0
    },
    {
        'name': 'Glucosa normal, comida grande',
        'glucoses': [100.0]*12,
        'carbs': 100.0
    },
    {
        'name': 'Anciana 93 años sedentaria - Glucosa 140 mg/dL',
        'glucoses': [140.0]*12,
        'carbs': 40.0,
        'basal_rate': 16.0,  # Lantus fijo
        'insulin_on_board': 0.0,
        'exercise_intensity': 0.0  # Sedentaria
    }
] 