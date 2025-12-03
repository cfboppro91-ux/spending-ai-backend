#ai-backend/model.py
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression


# ------------------- Helpers cơ bản -------------------

def parse_date(d: str) -> datetime:
    # d dạng: "YYYY-MM-DD"
    return datetime.strptime(d, "%Y-%m-%d")


def build_monthly_series(transactions: List[Dict[str, Any]]):
    """
    Gom giao dịch theo tháng:
    - key: YYYY-MM
    - value: { year, month, income, expense, per_category: {catId: expense} }
    """
    months: Dict[str, Dict[str, Any]] = {}

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


# ------------------- 1. Dự đoán chi tiêu tổng (theo tháng) -------------------

def predict_next_month_expense(series: List[Dict[str, Any]]):
    """
    Hồi quy tuyến tính chi tiêu theo tháng để đoán tháng tiếp theo.
    Nếu data < 3 tháng -> dùng trung bình.
    """
    if not series:
        return {
            "predicted_total_expense": 0.0,
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
                "income": float(m["income"]),
                "expense": float(m["expense"]),
            }
            for m in series
        ],
    }


# ------------------- 2. Thói quen chi tiêu -------------------

def top_categories_last_month(series: List[Dict[str, Any]], top_k: int = 3):
    """
    Lấy top category chi nhiều nhất ở tháng gần nhất.
    """
    if not series:
        return []

    last = series[-1]
    per_cat = last["per_category"]
    total = float(last["expense"]) or 1.0

    items = sorted(per_cat.items(), key=lambda x: x[1], reverse=True)
    res = [
        {
            "categoryId": cat_id,
            "expense": float(amount),
            "share": float(amount) / total,  # tỉ lệ % so với tổng chi tháng đó
        }
        for cat_id, amount in items[:top_k]
    ]
    return res


def spending_by_weekday(transactions: List[Dict[str, Any]]):
    """
    Tổng chi theo thứ trong tuần (0=Mon,..,6=Sun)
    """
    weekday_totals = defaultdict(float)
    labels = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]

    for t in transactions:
        if t.get("type") != "expense":
            continue
        try:
            dt = parse_date(t["date"])
        except Exception:
            continue
        weekday = dt.weekday()  # 0..6
        weekday_totals[weekday] += float(t.get("amount") or 0)

    stats = []
    for i in range(7):
        stats.append({
            "weekday": i,
            "label": labels[i],
            "expense": float(weekday_totals.get(i, 0.0)),
        })

    # tìm ngày chi nhiều nhất
    if stats:
        peak = max(stats, key=lambda x: x["expense"])
    else:
        peak = None

    return {
        "by_weekday": stats,
        "peak": peak,
    }


def spending_by_month_segment(transactions: List[Dict[str, Any]]):
    """
    Chia tháng thành 3 đoạn:
    - early: 1–10
    - mid:   11–20
    - late:  21–31
    """
    seg_totals = defaultdict(float)

    for t in transactions:
        if t.get("type") != "expense":
            continue
        try:
            dt = parse_date(t["date"])
        except Exception:
            continue
        day = dt.day
        if day <= 10:
            seg = "early"
        elif day <= 20:
            seg = "mid"
        else:
            seg = "late"
        seg_totals[seg] += float(t.get("amount") or 0)

    mapping_label = {
        "early": "Đầu tháng (1–10)",
        "mid":   "Giữa tháng (11–20)",
        "late":  "Cuối tháng (21–31)",
    }

    stats = []
    for key in ["early", "mid", "late"]:
        stats.append({
            "segment": key,
            "label": mapping_label[key],
            "expense": float(seg_totals.get(key, 0.0)),
        })

    # đoạn chi nhiều nhất
    if stats:
        peak = max(stats, key=lambda x: x["expense"])
    else:
        peak = None

    return {
        "by_segment": stats,
        "peak": peak,
    }


# ------------------- 3. Dự đoán tương lai tài chính -------------------

def compute_daily_projection(
    transactions: List[Dict[str, Any]],
    current_balance: Optional[float] = None,
    max_days: int = 60,
):
    """
    Gộp giao dịch theo ngày → tính:
    - avg_daily_income / expense / net
    - dự phóng expense 30 ngày
    - nếu có current_balance → dự phóng số dư tương lai
    """
    if not transactions:
        return {
            "avg_daily_income": 0.0,
            "avg_daily_expense": 0.0,
            "avg_daily_net": 0.0,
            "projected_30d_expense": 0.0,
            "projected_30d_balance": None if current_balance is None else float(current_balance),
        }

    # gom theo ngày
    day_map: Dict[str, Dict[str, float]] = {}

    for t in transactions:
        try:
            dt = parse_date(t["date"])
        except Exception:
            continue

        key = dt.strftime("%Y-%m-%d")
        if key not in day_map:
            day_map[key] = {"income": 0.0, "expense": 0.0}

        amount = float(t.get("amount") or 0)
        if t.get("type") == "income":
            day_map[key]["income"] += amount
        else:
            day_map[key]["expense"] += amount

    # sort theo ngày, chỉ lấy N ngày gần nhất
    sorted_days = sorted(day_map.keys())
    if len(sorted_days) > max_days:
        sorted_days = sorted_days[-max_days:]

    incomes = []
    expenses = []
    nets = []

    for d in sorted_days:
        inc = day_map[d]["income"]
        exp = day_map[d]["expense"]
        incomes.append(inc)
        expenses.append(exp)
        nets.append(inc - exp)

    avg_inc = float(np.mean(incomes)) if incomes else 0.0
    avg_exp = float(np.mean(expenses)) if expenses else 0.0
    avg_net = float(np.mean(nets)) if nets else 0.0

    proj_30_exp = max(0.0, avg_exp * 30.0)

    proj_30_balance = None
    if current_balance is not None:
        proj_30_balance = float(current_balance + avg_net * 30.0)

    return {
        "avg_daily_income": avg_inc,
        "avg_daily_expense": avg_exp,
        "avg_daily_net": avg_net,
        "projected_30d_expense": proj_30_exp,
        "projected_30d_balance": proj_30_balance,
        "days_used": len(sorted_days),
    }


# ------------------- 4. Gợi ý tiết kiệm (rule-based) -------------------

def generate_saving_tips(
    summary: Dict[str, Any],
    habits: Dict[str, Any],
    projection: Dict[str, Any],
):
    tips: List[str] = []

    total_income = float(summary.get("total_income", 0.0))
    total_expense = float(summary.get("total_expense", 0.0))
    months_count = int(summary.get("months_count", 0))

    # 1. Chi > thu trong nhiều tháng
    if months_count >= 2 and total_expense > total_income:
        tips.append(
            "Trong tổng dữ liệu, chi tiêu đang lớn hơn thu nhập. "
            "M nên đặt giới hạn chi hàng tháng hoặc tăng nguồn thu để tránh âm tiền."
        )

    # 2. Top category tháng gần nhất chi quá nhiều
    top_last = habits.get("top_categories_last_month") or []
    if top_last:
        top = top_last[0]
        share = float(top.get("share", 0.0))  # 0..1
        if share >= 0.4:
            tips.append(
                "Một nhóm chi tiêu chiếm hơn 40% tổng chi tháng gần nhất. "
                "M có thể xem lại các khoản trong nhóm này để cắt bớt những phần không cần thiết."
            )

    # 3. Cuối tuần chi nhiều
    weekday_info = habits.get("spending_by_weekday") or {}
    peak_wd = weekday_info.get("peak")
    if peak_wd and peak_wd.get("expense", 0.0) > 0:
        label = peak_wd.get("label", "")
        if "Thứ 7" in label or "Chủ nhật" in label:
            tips.append(
                f"Cuối tuần ({label}) là lúc m chi nhiều nhất. "
                "Thử đặt hạn mức chi cho cuối tuần, hoặc chọn các hoạt động ít tốn kém hơn."
            )

    # 4. Cuối tháng cháy túi
    seg_info = habits.get("spending_by_month_segment") or {}
    peak_seg = seg_info.get("peak")
    if peak_seg and peak_seg.get("segment") == "late" and peak_seg.get("expense", 0.0) > 0:
        tips.append(
            "Chi tiêu tập trung nhiều vào cuối tháng. "
            "M có thể phân bổ chi đều hơn trong tháng để tránh cuối tháng bị thiếu tiền."
        )

    # 5. Dòng tiền âm (burn rate)
    avg_net = float(projection.get("avg_daily_net", 0.0))
    if avg_net < 0:
        tips.append(
            "Trung bình mỗi ngày m đang chi nhiều hơn thu (dòng tiền âm). "
            "Thử đặt mục tiêu giảm 10–20% chi tiêu/ngày trong 1–2 tháng tới."
        )

    # Nếu không có tip nào
    if not tips:
        tips.append(
            "Thói quen chi tiêu hiện tại khá ổn định. "
            "M vẫn có thể tối ưu thêm bằng cách theo dõi kỹ các khoản lặt vặt và lên kế hoạch chi trước."
        )

    return tips


# ------------------- 5. Hàm chính cho API -------------------

def analyze_and_predict(
    transactions: List[Dict[str, Any]],
    current_balance: Optional[float] = None,
):
    """
    Input: list giao dịch
    Output: phân tích + dự đoán + gợi ý tiết kiệm
    """
    # series theo tháng
    series = build_monthly_series(transactions)

    # 1. Dự đoán chi tiêu tháng tới (từ chuỗi theo tháng)
    basic_pred = predict_next_month_expense(series)

    # 2. Top category tháng gần nhất
    top_cats = top_categories_last_month(series, top_k=3)

    # 3. Thói quen: theo thứ trong tuần, đoạn trong tháng
    weekday_info = spending_by_weekday(transactions)
    segment_info = spending_by_month_segment(transactions)

    # 4. Thống kê tổng
    total_income = sum(m["income"] for m in series) if series else 0.0
    total_expense = sum(m["expense"] for m in series) if series else 0.0

    summary = {
        "total_income": float(total_income),
        "total_expense": float(total_expense),
        "months_count": len(series),
    }

    # 5. Dự phóng tài chính theo ngày
    projection = compute_daily_projection(
        transactions,
        current_balance=current_balance,
    )

    # 6. Gợi ý tiết kiệm
    habits_struct = {
        "top_categories_last_month": top_cats,
        "spending_by_weekday": weekday_info,
        "spending_by_month_segment": segment_info,
    }
    saving_tips = generate_saving_tips(summary, habits_struct, projection)

    # Response cuối cùng
    return {
        "summary": summary,
        "prediction": {
            "next_month_expense": basic_pred["predicted_total_expense"],
            "method": basic_pred["method"],
            "history_months": basic_pred["history_months"],
        },
        "habits": habits_struct,
        "projection": projection,
        "saving_tips": saving_tips,
    }
