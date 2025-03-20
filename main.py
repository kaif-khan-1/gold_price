import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI()

# ✅ Try loading the model
try:
    model = joblib.load("gold_price_model.joblib")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ✅ Define input format
class InputData(BaseModel):
    SPX: float
    USO: float
    SLV: float
    EUR_USD: float

@app.get("/")
def home():
    return {"message": "Gold Price Prediction API is running!"}

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # ✅ Fix Pydantic V2 issue: Use .model_dump() instead of .dict()
        input_df = pd.DataFrame([data.model_dump()])
        
        prediction = model.predict(input_df)
        return {"predicted_price": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Run FastAPI with correct port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
