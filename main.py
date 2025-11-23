# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from model import analyze_and_predict

app = FastAPI(title="Spending AI API")


class Transaction(BaseModel):
    date: str
    amount: float
    type: str
    categoryId: str


class PredictRequest(BaseModel):
    transactions: List[Transaction]


@app.get("/")
def root():
    return {"msg": "Spending AI API OK"}


@app.post("/predict")
def predict_spending(body: PredictRequest):
    # chuyển sang list dict
    tx_list: List[Dict[str, Any]] = [t.dict() for t in body.transactions]
    result = analyze_and_predict(tx_list)
    return result
