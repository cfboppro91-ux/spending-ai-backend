# ai-backend/sample_loader.py
import os
from typing import List, Dict, Any

from sqlalchemy import create_engine, text

# Lấy connection string từ env, ví dụ:
# DATABASE_URL=postgresql+psycopg2://user:pass@host:port/dbname
DB_URL = os.environ.get("DATABASE_URL")

if not DB_URL:
    # tuỳ m, có thể raise hoặc để None
    raise RuntimeError("DATABASE_URL chưa được cấu hình cho ai-backend")

engine = create_engine(DB_URL, future=True)


def load_sample_transactions(limit: int | None = None) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu mẫu 6 tháng từ bảng transaction_12m.
    Trả về list dict: {date, amount, type, categoryId, id, description}
    """
    query = """
        SELECT date, amount, type, category_id, note
        FROM transactions_12m
        ORDER BY date
    """
    params = {}
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    txs: List[Dict[str, Any]] = []
    for r in rows:
        dt = r["date"]
        # dt: datetime/date -> convert "YYYY-MM-DD"
        date_str = dt.strftime("%Y-%m-%d")

        txs.append(
            {
                "date": date_str,
                "amount": float(r["amount"]),
                "type": r["type"],  # 'income' / 'expense'
                "categoryId": str(r["category_id"]) if r["category_id"] else None,
                "id": None,  # sample nên không cần id cụ thể
                "description": r.get("note"),
            }
        )

    return txs
