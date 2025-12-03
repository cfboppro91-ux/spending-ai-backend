#ai-backend/main.py
import os
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from model import analyze_and_predict
from model_v2 import analyze_and_predict_v2

# OpenAI client (lib mới)
from openai import OpenAI

app = FastAPI(title="Spending AI API")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


class Transaction(BaseModel):
    date: str
    amount: float
    type: str
    categoryId: str


class PredictRequest(BaseModel):
    transactions: List[Transaction]
    bank_transactions: Optional[List[Transaction]] = None
    current_balance: Optional[float] = None


class ChatRequest(BaseModel):
    question: str
    transactions: List[Transaction]
    bank_transactions: Optional[List[Transaction]] = None
    current_balance: Optional[float] = None


@app.get("/")
def root():
    return {"msg": "Spending AI API OK"}


@app.post("/predict")
def predict_spending(body: PredictRequest):
    tx_list = [t.dict() for t in body.transactions]
    bank_tx_list = [t.dict() for t in (body.bank_transactions or [])]
    result = analyze_and_predict_v2(tx_list, bank_transactions=bank_tx_list, current_balance=body.current_balance)
    return result

@app.post("/chat")
def chat_spending_assistant(body: ChatRequest):
    tx_list = [t.dict() for t in body.transactions]
    bank_tx_list = [t.dict() for t in (body.bank_transactions or [])]
    analysis = analyze_and_predict_v2(tx_list, bank_transactions=bank_tx_list, current_balance=body.current_balance)

    # prepare a brief structured summary to give to LLM (max N chars)
    summary = {
        "total_income": analysis["summary"]["total_income"],
        "total_expense": analysis["summary"]["total_expense"],
        "months_count": analysis["summary"]["months_count"],
        "next_month_pred": analysis["prediction"]["predicted"],
        "next_month_ci": analysis["prediction"]["conf_interval"],
        "top_categories_last_month": analysis["habits"]["top_categories_last_month"],
    }

    system_prompt = (
        "Bạn là trợ lý tài chính cá nhân, trả lời ngắn gọn, thân thiện, tiếng Việt.\n"
        "KHI TRẢ LỜI: luôn **trích dẫn số liệu** \n"
        "Không bịa số liệu ngoài JSON tớ gửi.\n"
        "Đề xuất hành động ngắn gọn 1-2 bước.\n"
    )

    user_prompt = (
        f"Câu hỏi: {body.question}\n\n"
        f"Tóm tắt số liệu: {json.dumps(summary, ensure_ascii=False)}\n\n"
        "Trả lời ngắn gọn, dễ hiểu."
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    return {"answer": answer, "analysis": analysis}
