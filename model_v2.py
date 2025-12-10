#ai-backend/model_v2.py
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from joblib import dump, load
import math
from typing import Set

GLOBAL_BASELINE_PATH = "models/global_baseline.joblib"
GLOBAL_BASELINE = None  # sẽ là dict: {"reg": ..., "mean": ..., "months": int}


def train_global_baseline(transactions: List[Dict[str, Any]]):
    """
    Train global model từ dữ liệu mẫu (vd: transaction_12m).
    Lưu lại slope + intercept + mean để dùng cho user sau này.
    """
    global GLOBAL_BASELINE

    series = build_monthly_series(transactions)
    if not series:
        GLOBAL_BASELINE = None
        return

    y = np.array([m["expense"] for m in series], dtype=float)
    X = np.arange(len(y)).reshape(-1, 1)

    baseline: Dict[str, Any] = {
        "mean": float(y.mean()),
        "months": int(len(y)),
    }

    if len(y) >= 3:
        reg = LinearRegression()
        reg.fit(X, y)
        baseline["reg_coef"] = float(reg.coef_[0])
        baseline["reg_intercept"] = float(reg.intercept_)
    else:
        baseline["reg_coef"] = None
        baseline["reg_intercept"] = None

    GLOBAL_BASELINE = baseline
    # lưu ra file
    os.makedirs("models", exist_ok=True)
    dump(baseline, GLOBAL_BASELINE_PATH)


def load_global_baseline():
    global GLOBAL_BASELINE
    if GLOBAL_BASELINE is not None:
        return GLOBAL_BASELINE
    try:
        baseline = load(GLOBAL_BASELINE_PATH)
        GLOBAL_BASELINE = baseline
        return baseline
    except Exception:
        GLOBAL_BASELINE = None
        return None


def predict_from_global_baseline() -> Optional[float]:
    """
    Dự đoán chi tháng sau theo model global (không có data user).
    """
    baseline = load_global_baseline()
    if not baseline:
        return None

    months = baseline.get("months", 0)
    mean = baseline.get("mean", 0.0)
    coef = baseline.get("reg_coef")
    intercept = baseline.get("reg_intercept")

    if coef is not None and intercept is not None and months > 0:
        # dự đoán cho tháng tiếp theo sau chuỗi global
        x_next = np.array([[months]])  # index = số tháng hiện có
        pred = float(coef * x_next[0][0] + intercept)
        return max(0.0, pred)
    else:
        return max(0.0, float(mean))

def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")

def build_monthly_series(transactions: List[Dict[str,Any]]):
    months = {}
    for t in transactions:
        try:
            dt = parse_date(t["date"])
        except Exception:
            continue
        key = f"{dt.year}-{dt.month:02d}"
        if key not in months:
            months[key] = {"year": dt.year, "month": dt.month, "income":0.0, "expense":0.0, "per_category":defaultdict(float)}
        amt = float(t.get("amount") or 0)
        ttype = t.get("type","expense")
        cat = t.get("categoryId","unknown")
        if ttype == "income":
            months[key]["income"] += amt
        else:
            months[key]["expense"] += amt
            months[key]["per_category"][cat] += amt
    keys = sorted(months.keys())
    return [months[k] for k in keys]

def linear_predict(y: np.ndarray) -> Tuple[float, float]:
    # returns (pred, std_residual)
    if len(y) == 0:
        return 0.0, 0.0
    X = np.arange(len(y)).reshape(-1,1)
    if len(y) >= 3:
        m = LinearRegression()
        m.fit(X,y)
        pred = float(m.predict([[len(y)]])[0])
        residuals = y - m.predict(X)
        std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals))
        return max(0.0, pred), std
    else:
        mean = float(np.mean(y))
        std = float(np.std(y)) if len(y)>1 else 0.0
        return mean, std

def predict_next_month_expense_v2(series: List[Dict[str,Any]]):
    """
    - Nếu user có >=3 tháng: dùng thuần data user (linear_regression / mean_fallback như cũ).
    - Nếu user có 0–2 tháng: blend giữa global model (học từ 6 tháng mẫu)
      và user data.
    """
    if not series:
        # không có data user -> thử dùng global model
        global_pred = predict_from_global_baseline()
        if global_pred is not None:
            return {
                "predicted": global_pred,
                "method": "global_only",
                "conf_interval": [global_pred, global_pred],
                "history": [],
            }
        return {"predicted": 0.0, "method": "no_data", "conf_interval": [0.0, 0.0], "history": []}

    y = np.array([m["expense"] for m in series], dtype=float)
    user_months = len(y)

    # user-based prediction (giống cũ)
    user_pred, user_std = linear_predict(y)
    base_method = "linear_regression" if user_months >= 3 else "mean_fallback"

    global_pred = predict_from_global_baseline()

    # Nếu đủ 3 tháng -> dùng user 100%
    if user_months >= 3 or global_pred is None:
        lo = max(0.0, user_pred - 1.96 * (user_std or 0.0))
        hi = user_pred + 1.96 * (user_std or 0.0)
        return {
            "predicted": user_pred,
            "method": base_method,
            "conf_interval": [lo, hi],
            "history": [
                {
                    "year": m["year"],
                    "month": m["month"],
                    "expense": float(m["expense"]),
                    "income": float(m["income"]),
                }
                for m in series
            ],
        }

    # ---- Blend global + user (0–2 tháng user) ----
    # weight theo số tháng user: 0 -> 0, 1 -> 1/3, 2 -> 2/3
    alpha = min(1.0, user_months / 3.0)
    blended_pred = float(alpha * user_pred + (1.0 - alpha) * global_pred)

    # std tạm thời dùng user_std cho đơn giản
    lo = max(0.0, blended_pred - 1.96 * (user_std or 0.0))
    hi = blended_pred + 1.96 * (user_std or 0.0)

    return {
        "predicted": blended_pred,
        "method": "global_user_blend",
        "conf_interval": [lo, hi],
        "history": [
            {
                "year": m["year"],
                "month": m["month"],
                "expense": float(m["expense"]),
                "income": float(m["income"]),
            }
            for m in series
        ],
    }

def per_category_forecast(series: List[Dict[str,Any]]):
    # build monthly per-category series and run linear fit per category
    # output: {cat: {pred, method, conf_interval}}
    # aggregate months index
    months = [f'{m["year"]}-{m["month"]:02d}' for m in series]
    cat_map = defaultdict(list)  # cat -> list aligned with months
    for m in series:
        for cat, amt in m["per_category"].items():
            cat_map[cat].append((f'{m["year"]}-{m["month"]:02d}', float(amt)))
    res = {}
    for cat, pairs in cat_map.items():
        # align to full months (fill 0 if missing)
        month_to_amt = {k:v for k,v in pairs}
        y = np.array([month_to_amt.get(mon, 0.0) for mon in months], dtype=float)
        pred, std = linear_predict(y)
        lo = max(0.0, pred - 1.96 * (std or 0.0))
        hi = pred + 1.96 * (std or 0.0)
        res[cat] = {"predicted":pred, "conf_interval":[lo,hi], "method": "linear_regression" if len(y)>=3 else "mean_fallback"}
    return res

def detect_anomalies(transactions: List[Dict[str,Any]]):
    # simple: z-score on amounts for expense and income separately, fallback to IsolationForest
    amounts = np.array([float(t.get("amount") or 0) for t in transactions])
    if len(amounts) >= 6:
        mean = amounts.mean()
        std = amounts.std(ddof=1) if len(amounts) > 1 else 0.0
        anomalies = []
        if std > 0:
            for t in transactions:
                z = (float(t.get("amount") or 0) - mean)/std
                if abs(z) > 3.0:
                    anomalies.append({"id": t.get("id"), "date": t.get("date"), "amount": t.get("amount"), "z": float(z)})
            if anomalies:
                return anomalies
    # fallback IsolationForest (works on amounts)
    try:
        amounts_reshaped = amounts.reshape(-1,1)
        iso = IsolationForest(contamination=0.03, random_state=42)
        iso.fit(amounts_reshaped)
        preds = iso.predict(amounts_reshaped)  # -1 anomaly
        anomalies = []
        for i, p in enumerate(preds):
            if p == -1:
                t = transactions[i]
                anomalies.append({"id": t.get("id"), "date": t.get("date"), "amount": t.get("amount")})
        return anomalies
    except Exception:
        return []

def compute_daily_projection(transactions, current_balance=None, max_days=60):
    # reuse simple average daily net approach (same as your old code)
    # ... keep as before (omitted for brevity) or import if you prefer
    # quick re-use small impl:
    day_map={}
    for t in transactions:
        try:
            dt = parse_date(t["date"])
        except Exception:
            continue
        key = dt.strftime("%Y-%m-%d")
        day_map.setdefault(key, {"income":0.0,"expense":0.0})
        amt = float(t.get("amount") or 0)
        if t.get("type")=="income":
            day_map[key]["income"] += amt
        else:
            day_map[key]["expense"] += amt
    sorted_days = sorted(day_map.keys())
    if len(sorted_days) > max_days:
        sorted_days = sorted_days[-max_days:]
    incomes = [day_map[d]["income"] for d in sorted_days]
    expenses = [day_map[d]["expense"] for d in sorted_days]
    import numpy as _np
    avg_inc = float(_np.mean(incomes)) if incomes else 0.0
    avg_exp = float(_np.mean(expenses)) if expenses else 0.0
    avg_net = avg_inc - avg_exp
    proj_30_exp = max(0.0, avg_exp * 30)
    proj_bal = None
    if current_balance is not None:
        proj_bal = float(current_balance + avg_net * 30)
    return {"avg_daily_income":avg_inc,"avg_daily_expense":avg_exp,"avg_daily_net":avg_net,"projected_30d_expense":proj_30_exp,"projected_30d_balance":proj_bal,"days_used":len(sorted_days)}

def generate_saving_tips(summary, habits, projection):
    # you can reuse original rules; simplified here
    tips=[]
    total_income = summary.get("total_income",0.0)
    total_expense = summary.get("total_expense",0.0)
    months_count = summary.get("months_count",0)
    if months_count>=2 and total_expense>total_income:
        tips.append("Chi vượt thu: cân nhắc giảm chi tiêu hoặc tìm nguồn thu thêm.")
    top = (habits.get("top_categories_last_month") or [])
    if top and top[0].get("share",0) >= 0.4:
        tips.append("Một nhóm chiếm >40% chi tháng gần nhất, xem lại nhóm đó.")
    if not tips:
        tips.append("Tình hình ổn, tiếp tục theo dõi.")
    return tips

def analyze_and_predict_v2(
    transactions: List[Dict[str,Any]],
    bank_transactions: Optional[List[Dict[str,Any]]] = None,
    current_balance: Optional[float] = None
):
    # ====== MERGE DATA ======
    bank_tx = bank_transactions or []
    merged = merge_transactions(transactions, bank_tx)

    # ====== BUILD MONTHLY SERIES ======
    series = build_monthly_series(merged)
    basic = predict_next_month_expense_v2(series)
    per_cat = per_category_forecast(series)

    # ====== HABITS ======
    top_struct = []
    if series:
        last = series[-1]
        total = last["expense"] or 1
        for cat, amt in sorted(last["per_category"].items(), key=lambda x: x[1], reverse=True)[:3]:
            top_struct.append({
                "categoryId": cat,
                "expense": float(amt),
                "share": float(amt) / total
            })

    habits = {
        "top_categories_last_month": top_struct
    }

    # ====== DAILY PROJECTION ======
    projection = compute_daily_projection(merged, current_balance=current_balance)

    # ====== ANOMALIES ======
    anomalies = detect_anomalies(merged)

    # ====== SUMMARY ======
    summary = {
        "total_income": float(sum(m["income"] for m in series)),
        "total_expense": float(sum(m["expense"] for m in series)),
        "months_count": len(series)
    }

    # ====== SAVING TIPS ======
    saving_tips = generate_saving_tips(summary, habits, projection)

    # ====== FINAL RESPONSE ======
    return {
        "summary": summary,
        "prediction": basic,
        "per_category_prediction": per_cat,
        "habits": habits,
        "projection": projection,
        "anomalies": anomalies,
        "saving_tips": saving_tips,
        "raw_monthly_series": series,
        "meta": {
            "merged_count": len(merged),
            "app_count": len(transactions),
            "bank_count": len(bank_tx),
        }
    }

# optional small persistence
def save_model(obj, path="models/ai_model.joblib"):
    dump(obj, path)

def load_model(path="models/ai_model.joblib"):
    return load(path)

def normalize_tx_key(t: Dict[str,Any]) -> str:
    # tạo key đơn giản để de-dup: date|amount|type|shortdesc
    date = (t.get("date") or "")[:10]
    amount = str(round(float(t.get("amount") or 0), 2))
    ttype = t.get("type") or "expense"
    desc = (t.get("description") or t.get("note") or "")[:40].strip().lower()
    cat = t.get("categoryId") or ""
    return f"{date}|{amount}|{ttype}|{desc}|{cat}"

def merge_transactions(app_tx: List[Dict[str,Any]], bank_tx: List[Dict[str,Any]]):
    """
    - Gộp 2 nguồn
    - De-dup bằng normalize_tx_key
    - Giữ cả source field: 'app' or 'bank'
    - Nếu trùng, ưu tiên app (giả sử app có category mapping tốt hơn)
    """
    out = []
    seen: Set[str] = set()

    # mark source
    for t in app_tx:
        t2 = dict(t)
        t2["_source"] = "app"
        key = normalize_tx_key(t2)
        seen.add(key)
        out.append(t2)

    for t in bank_tx:
        t2 = dict(t)
        t2["_source"] = "bank"
        key = normalize_tx_key(t2)
        if key in seen:
            # skip bank duplicate
            continue
        seen.add(key)
        out.append(t2)

    # optional: sort by date asc
    try:
        out.sort(key=lambda x: x.get("date") or "")
    except Exception:
        pass
    return out