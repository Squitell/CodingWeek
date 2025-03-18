from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("XGBoost_model_20250318_1312.pkl")

# Define Input Schema (Must Match Form.js Exactly)
class PatientData(BaseModel):
    Recipientgender: int
    Stemcellsource: int
    Donorage: float
    Donorage35: int
    IIIV: int
    Gendermatch: int
    DonorABO: int
    RecipientABO: int
    RecipientRh: int
    ABOmatch: int
    CMVstatus: int
    DonorCMV: int
    RecipientCMV: int
    Disease: str
    Riskgroup: int
    Txpostrelapse: int
    Diseasegroup: int
    HLAmatch: int
    HLAmismatch: int
    Antigen: int
    Alel: int
    Recipientage: float
    Recipientage10: int
    Relapse: int
    aGvHDIIIIV: int
    extcGvHD: int
    CD34kgx10d6: float
    CD3dCD34: float
    CD3dkgx10d8: float
    Rbodymass: float
    ANCrecovery: float
    PLTrecovery: float
    survival_time: float  # NEW FIELD
    
disease_mapping = {
    "ALL": 0,
    "AML": 1,
    "chronic": 2,
    "nonmalignant": 3,
    "lymphoma": 4
}    

@app.post("/predict")
async def predict(data: PatientData):
    disease_encoded = disease_mapping.get(data.Disease, -1)

    # Correctly retrieve the model's feature names without using 'or' on arrays.
    if hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_
    elif hasattr(model, "feature_names_in"):
        feature_names = model.feature_names_in
    else:
        return {"error": "Model does not have feature names attribute."}

    # Convert input to DataFrame using the obtained feature names
    input_data = pd.DataFrame([[
        int(data.Recipientgender), int(data.Stemcellsource), float(data.Donorage), int(data.Donorage35),
        int(data.IIIV), int(data.Gendermatch), int(data.DonorABO), int(data.RecipientABO), int(data.RecipientRh),
        int(data.ABOmatch), int(data.CMVstatus), int(data.DonorCMV), int(data.RecipientCMV), int(disease_encoded),
        int(data.Riskgroup), int(data.Txpostrelapse), int(data.Diseasegroup), int(data.HLAmatch),
        int(data.HLAmismatch), int(data.Antigen), int(data.Alel), float(data.Recipientage), int(data.Recipientage10),
        int(data.Relapse), int(data.aGvHDIIIIV), int(data.extcGvHD), float(data.CD34kgx10d6), float(data.CD3dCD34),
        float(data.CD3dkgx10d8), float(data.Rbodymass), float(data.ANCrecovery), float(data.PLTrecovery),
        float(data.survival_time)
    ]], columns=feature_names)

    # Validate feature count
    expected_features = model.n_features_in_
    print(f"Received {input_data.shape[1]} features, Model expects {expected_features}")
    
    if input_data.shape[1] != expected_features:
        return {"error": f"Expected {expected_features} features but received {input_data.shape[1]}"}
    
    prediction_raw = model.predict(input_data)
    print(f"Raw Model Prediction: {prediction_raw}")

    prediction = int(prediction_raw[0])
    result = "Survive" if prediction == 1 else "Not Survive" if prediction == 0 else "❌ Error: Unexpected Prediction Value"

    print(f"Model Raw Prediction: {prediction_raw}")
    print(f"Final API Response: {result}")

    return {"prediction": result}

    
    
if __name__ == "__main__":
    print("Starting backend server with uvicorn...")
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


