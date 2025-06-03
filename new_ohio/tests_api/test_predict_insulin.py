import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import numpy as np
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from new_ohio.api.predict_insulin import app, prepare_observation

client = TestClient(app)

def test_prepare_observation():
    """Test the observation preparation function"""
    cgm_values = [120, 125, 130]
    hour_of_day = 15
    carb_input = 30
    insulin_on_board = 2.5
    
    obs = prepare_observation(cgm_values, hour_of_day, carb_input, insulin_on_board)
    
    # Check shape
    assert len(obs) == 28  # 24 CGM values + 4 other features
    
    # Check CGM values are padded correctly
    assert len([x for x in obs[:24] if x != 0]) == 3  # Only 3 non-zero values
    
    # Check other features using np.isclose for floating point comparison
    assert np.isclose(obs[24], hour_of_day / 24.0)  # Normalized hour
    assert np.isclose(obs[25], 0.0)  # bolus_log
    assert np.isclose(obs[26], np.log1p(carb_input))  # carb_log
    assert np.isclose(obs[27], np.log1p(insulin_on_board))  # iob_log

def test_predict_insulin_valid_request():
    """Test the endpoint with valid data"""
    current_time = datetime.now().isoformat()
    response = client.post(
        "/predict_insulin",
        json={
            "date": current_time,
            "cgmPrev": [120, 125, 130, 135, 140],
            "glucoseObjective": 100,
            "carbs": 30,
            "insulinOnBoard": 2.5,
            "sleepLevel": 7,
            "workLevel": 5,
            "activityLevel": 3
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "total" in data
    assert "breakdown" in data
    assert "correctionDose" in data["breakdown"]
    assert "mealDose" in data["breakdown"]
    assert "activityAdjustment" in data["breakdown"]
    assert "timeAdjustment" in data["breakdown"]
    
    # Check value types
    assert isinstance(data["total"], float)
    assert isinstance(data["breakdown"]["correctionDose"], float)
    assert isinstance(data["breakdown"]["mealDose"], float)
    assert isinstance(data["breakdown"]["activityAdjustment"], float)
    assert isinstance(data["breakdown"]["timeAdjustment"], float)

def test_predict_insulin_invalid_cgm():
    """Test the endpoint with invalid CGM values"""
    current_time = datetime.now().isoformat()
    response = client.post(
        "/predict_insulin",
        json={
            "date": current_time,
            "cgmPrev": [],  # Empty CGM values
            "glucoseObjective": 100,
            "carbs": 30,
            "insulinOnBoard": 2.5,
            "sleepLevel": 7,
            "workLevel": 5,
            "activityLevel": 3
        }
    )
    
    assert response.status_code == 200  # Should still work as we handle empty arrays

def test_predict_insulin_invalid_date():
    """Test the endpoint with invalid date format"""
    response = client.post(
        "/predict_insulin",
        json={
            "date": "invalid-date",  # Invalid date format
            "cgmPrev": [120, 125, 130],
            "glucoseObjective": 100,
            "carbs": 30,
            "insulinOnBoard": 2.5,
            "sleepLevel": 7,
            "workLevel": 5,
            "activityLevel": 3
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_predict_insulin_missing_fields():
    """Test the endpoint with missing required fields"""
    response = client.post(
        "/predict_insulin",
        json={
            "date": datetime.now().isoformat(),
            "cgmPrev": [120, 125, 130],
            # Missing other required fields
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_predict_insulin_invalid_levels():
    """Test the endpoint with invalid level values"""
    current_time = datetime.now().isoformat()
    response = client.post(
        "/predict_insulin",
        json={
            "date": current_time,
            "cgmPrev": [120, 125, 130],
            "glucoseObjective": 100,
            "carbs": 30,
            "insulinOnBoard": 2.5,
            "sleepLevel": 11,  # Invalid level (should be 1-10)
            "workLevel": 5,
            "activityLevel": 3
        }
    )
    
    assert response.status_code == 200  # Should still work as we don't validate these ranges

def test_predict_insulin_edge_cases():
    """Test the endpoint with edge case values"""
    current_time = datetime.now().isoformat()
    
    # Test with maximum CGM values
    response = client.post(
        "/predict_insulin",
        json={
            "date": current_time,
            "cgmPrev": [400] * 24,  # Maximum CGM values
            "glucoseObjective": 100,
            "carbs": 30,
            "insulinOnBoard": 2.5,
            "sleepLevel": 7,
            "workLevel": 5,
            "activityLevel": 3
        }
    )
    
    assert response.status_code == 200
    
    # Test with minimum values
    response = client.post(
        "/predict_insulin",
        json={
            "date": current_time,
            "cgmPrev": [40],  # Minimum CGM value
            "glucoseObjective": 100,
            "carbs": 0,
            "insulinOnBoard": 0,
            "sleepLevel": 1,
            "workLevel": 1,
            "activityLevel": 1
        }
    )
    
    assert response.status_code == 200 