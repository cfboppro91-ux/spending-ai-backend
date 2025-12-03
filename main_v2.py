# ai-backend/main_v2.py (snippet)
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from model_v2 import V2Model, MODEL_PATH, predict_all
import joblib, os

app = FastAPI()

class PredictV2Request(BaseModel):
    user_id: str | None = None
    transactions: list[dict] | None = None
    current_balance: float | None = 0.0

@app.post("/ai/v2/predict")
def ai_v2_predict(body: PredictV2Request):
    # if user has a saved per-user model, load it; else load global model
    model_path = os.environ.get('AI_MODEL_DIR','./ai_models')
    user_path = None
    if body.user_id:
        maybe = os.path.join(model_path, f"model_v2_{body.user_id}.joblib")
        if os.path.exists(maybe):
            user_path = maybe
    if user_path:
        model = V2Model.load(user_path)
    else:
        model = V2Model.load(os.path.join(model_path, 'model_v2_global.joblib'))
    txs = body.transactions or []
    res = predict_all(model, txs, body.current_balance or 0.0)
    return res
