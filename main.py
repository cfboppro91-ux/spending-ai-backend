# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from model import analyze_and_predict

app = FastAPI(title="Spending AI API")


class Transaction(BaseModel):
    date: str
    amount: float
    type: str
    categoryId: str


class PredictRequest(BaseModel):
    transactions: List[Transaction]
    # số dư hiện tại (nếu app gửi lên) để dự đoán tương lai tài chính
    current_balance: Optional[float] = None


@app.get("/")
def root():
    return {"msg": "Spending AI API OK"}


@app.post("/predict")
def predict_spending(body: PredictRequest):
    tx_list: List[Dict[str, Any]] = [t.dict() for t in body.transactions]
    result = analyze_and_predict(
        tx_list,
        current_balance=body.current_balance
    )
    return result
