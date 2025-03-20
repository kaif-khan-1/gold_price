import os
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load trained model
model = joblib.load("gold_price_model.joblib")

# Define input format
class InputData(BaseModel):
    SPX: float
    USO: float
    SLV: float
    EUR_USD: float

@app.get("/")
def home():
    return {"message": "Gold Price Prediction API is running!"}

@app.post("/predict/")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"predicted_price": prediction[0]}

# Use correct PORT variable from Railway
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
