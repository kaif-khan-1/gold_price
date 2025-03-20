import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI()

# ✅ Enable CORS for all origins (Allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Change this to your React domain for security
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all HTTP methods
    allow_headers=["*"],  # ✅ Allow all headers
)

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
    EUR_USD: float

@app.get("/")
def home():
    return {"message": "Gold Price Prediction API is running!"}

@app.post("/predict/")
async def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        input_df.rename(columns={"EUR_USD": "EUR/USD"}, inplace=True)

        prediction = model.predict(input_df)
        return {"predicted_price": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Run FastAPI with correct Railway port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
