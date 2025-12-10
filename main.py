import os
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# sử dụng model_v2 (hỗ trợ bank tx merge)
from model_v2 import analyze_and_predict_v2, build_monthly_series, train_global_baseline
from sample_loader import load_sample_transactions
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
    categoryId: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None


class PredictRequest(BaseModel):
    transactions: List[Transaction]
    bank_transactions: Optional[List[Transaction]] = None
    current_balance: Optional[float] = None


class ChatRequest(BaseModel):
    question: str
    transactions: List[Transaction]
    bank_transactions: Optional[List[Transaction]] = None
    current_balance: Optional[float] = None

class TrainRequest(BaseModel):
    transactions: List[Transaction]

@app.get("/")
def root():
    return {"msg": "Spending AI API OK"}


@app.post("/predict")
def predict_spending(body: PredictRequest):
    # Data user gửi lên
    tx_list: List[Dict[str, Any]] = [t.dict() for t in body.transactions]
    bank_list: List[Dict[str, Any]] = [t.dict() for t in (body.bank_transactions or [])]

    # Xem user hiện có bao nhiêu tháng data
    user_series = build_monthly_series(tx_list)
    user_months = len(user_series)

    # Mặc định: chỉ dùng data user
    final_tx = tx_list
    sample_used = False

    # Nếu user < 3 tháng -> gộp thêm 6 tháng mẫu
    if user_months < 3:
        sample_tx = load_sample_transactions()
        final_tx = sample_tx + tx_list
        sample_used = True

    result = analyze_and_predict_v2(
        final_tx,
        bank_transactions=bank_list,
        current_balance=body.current_balance,
    )

    # chỉnh lại meta cho dễ debug
    meta = result.get("meta", {})
    meta["user_tx_count"] = len(tx_list)
    meta["used_sample_12m"] = sample_used
    meta["user_months"] = user_months
    result["meta"] = meta

    return result


@app.post("/chat")
def chat_spending_assistant(body: ChatRequest):
    """
    Trợ lý tài chính:
    - Nhận câu hỏi tự nhiên (ăn uống, giải trí…)
    - Có dữ liệu chi tiêu + số dư ước tính (bao gồm cả bank_transactions)
    - Dùng OpenAI để trả lời.
    """
    tx_list: List[Dict[str, Any]] = [t.dict() for t in body.transactions]
    bank_list: List[Dict[str, Any]] = [t.dict() for t in (body.bank_transactions or [])]

    # phân tích lại bằng model_v2 (gộp app + bank)
    analysis = analyze_and_predict_v2(
        tx_list,
        bank_transactions=bank_list,
        current_balance=body.current_balance
    )

    # Gom context gửi cho LLM
    context_json = json.dumps(analysis, ensure_ascii=False)

    system_prompt = (
        "Bạn là trợ lý tài chính cá nhân, trả lời bằng tiếng Việt, thân thiện, ngắn gọn.\n"
        "Bạn sẽ nhận được:\n"
        "- Một câu hỏi của người dùng về chi tiêu / ăn uống / giải trí / kế hoạch tiền bạc.\n"
        "- Một JSON chứa phân tích chi tiêu: tổng thu/chi, dự đoán tháng sau, thói quen chi tiêu, gợi ý tiết kiệm.\n\n"
        "Nhiệm vụ:\n"
        "1. Trả lời đúng trọng tâm câu hỏi.\n"
        "2. Dựa trên số liệu trong JSON để đưa ví dụ cụ thể (số tiền, xu hướng... nếu phù hợp).\n"
        "3. Không bịa số liệu ngoài những gì JSON cung cấp.\n"
        "4. Không nói về JSON, chỉ nói như đang hiểu rõ tình hình tài chính của người dùng.\n"
    )

    user_prompt = (
        f"Câu hỏi của người dùng: {body.question}\n\n"
        f"Dữ liệu phân tích chi tiêu (JSON):\n{context_json}"
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "analysis": analysis,
    }

@app.post("/train/global")
def train_global(body: TrainRequest):
    """
    Train global baseline model từ dữ liệu mẫu (vd: transaction_12m).
    Gọi endpoint này 1 lần từ backend chính (hoặc script) với 6 tháng mẫu.
    """
    tx_list: List[Dict[str, Any]] = [t.dict() for t in body.transactions]

    train_global_baseline(tx_list)

    return {"msg": "trained_global_baseline", "count": len(tx_list)}