# model.py
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from sklearn.linear_model import LinearRegression


def parse_date(d: str) -> datetime:
    # d dạng: "YYYY-MM-DD"
    return datetime.strptime(d, "%Y-%m-%d")


def build_monthly_series(transactions: List[Dict[str, Any]]):
    """
    Gom giao dịch theo tháng:
    - key: YYYY-MM
    - value: { income: ..., expense: ..., per_category: {catId: expense} }
    """
    months = {}
    for t in transactions:
        try:
          dt = parse_date(t["date"])
        except Exception:
          continue

        key = f"{dt.year}-{dt.month:02d}"
        if key not in months:
            months[key] = {
                "year": dt.year,
                "month": dt.month,
                "income": 0.0,
                "expense": 0.0,
                "per_category": defaultdict(float),
            }

        amount = float(t.get("amount") or 0)
        ttype = t.get("type", "expense")
        cat_id = t.get("categoryId", "unknown")

        if ttype == "income":
            months[key]["income"] += amount
        else:
            months[key]["expense"] += amount
            months[key]["per_category"][cat_id] += amount

    # sort theo thời gian
    sorted_keys = sorted(months.keys())
    series = [months[k] for k in sorted_keys]
    return series


def predict_next_month_expense(series: List[Dict[str, Any]]):
    """
    Hồi quy tuyến tính chi tiêu theo tháng để đoán tháng tiếp theo.
    Nếu data < 3 tháng -> dùng trung bình.
    """
    if not series:
        return {
            "predicted_total_expense": 0,
            "method": "no_data",
        }

    X = np.arange(len(series)).reshape(-1, 1)   # 0,1,2,...
    y = np.array([m["expense"] for m in series], dtype=float)

    if len(series) >= 3:
        model = LinearRegression()
        model.fit(X, y)
        next_index = np.array([[len(series)]])
        pred = float(model.predict(next_index)[0])
        method = "linear_regression"
    else:
        pred = float(y.mean())
        method = "mean_fallback"

    if pred < 0:
        pred = 0.0

    return {
        "predicted_total_expense": pred,
        "method": method,
        "history_months": [
            {
                "year": m["year"],
                "month": m["month"],
                "income": m["income"],
                "expense": m["expense"],
            }
            for m in series
        ],
    }


def top_categories_last_month(series: List[Dict[str, Any]], top_k: int = 3):
    """
    Lấy top category chi nhiều nhất ở tháng gần nhất.
    """
    if not series:
        return []

    last = series[-1]
    per_cat = last["per_category"]
    items = sorted(per_cat.items(), key=lambda x: x[1], reverse=True)
    res = [
        {
            "categoryId": cat_id,
            "expense": float(amount),
        }
        for cat_id, amount in items[:top_k]
    ]
    return res


def analyze_and_predict(transactions: List[Dict[str, Any]]):
    """
    Hàm chính: input là list giao dịch,
    output: phân tích + dự đoán trả về app.
    """
    series = build_monthly_series(transactions)
    basic_pred = predict_next_month_expense(series)
    top_cats = top_categories_last_month(series, top_k=3)

    total_income = sum(m["income"] for m in series)
    total_expense = sum(m["expense"] for m in series)

    return {
        "summary": {
            "total_income": total_income,
            "total_expense": total_expense,
            "months_count": len(series),
        },
        "prediction": {
            "next_month_expense": basic_pred["predicted_total_expense"],
            "method": basic_pred["method"],
        },
        "top_categories_last_month": top_cats,
    }
