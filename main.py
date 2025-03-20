import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI()

# ✅ Load the trained ML model
try:
    model = joblib.load("gold_price_model.joblib")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ✅ Define input format
class InputData(BaseModel):
    SPX: float
    USO: float
    SLV: float
    EUR_USD: float  # ✅ Fix: Keep API name consistent

@app.get("/")
def home():
    return {"message": "Gold Price Prediction API is running!"}

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # ✅ Convert API input to DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # ✅ Fix: Rename column to match trained model
        input_df.rename(columns={"EUR_USD": "EUR/USD"}, inplace=True)

        # ✅ Make prediction
        prediction = model.predict(input_df)
        return {"predicted_price": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Run FastAPI with correct Railway port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
