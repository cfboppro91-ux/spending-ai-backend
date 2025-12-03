# ai-backend/train_model.py
import os
import json
import psycopg2
from datetime import date
from model_v2 import train_model_from_transactions, V2Model, MODEL_PATH

DB_DSN = os.environ.get('DATABASE_URL')  # postgres connection string
OUTPUT_PATH = os.environ.get('AI_MODEL_DIR','./ai_models')

def fetch_user_transactions(conn, user_id, months=24):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT date::date, amount, type, COALESCE(category_id, 'unknown') as category_id
            FROM transactions_12m
            WHERE user_id = %s
              AND date >= (current_date - interval '%s months')
            ORDER BY date ASC
        """, (str(user_id), months))
        rows = cur.fetchall()
    txs = []
    for r in rows:
        txs.append({
            'date': r[0].isoformat(),
            'amount': float(r[1]),
            'type': r[2],
            'categoryId': r[3],
        })
    return txs

def main_train_all():
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        # get users that have transactions
        cur.execute("SELECT DISTINCT user_id FROM transactions_12m")
        users = [r[0] for r in cur.fetchall()]
    for u in users:
        print("Training for user", u)
        txs = fetch_user_transactions(conn, u)
        if not txs:
            continue
        model = train_model_from_transactions(txs)
        # save by user id
        user_model_path = os.path.join(OUTPUT_PATH, f"model_v2_{u}.joblib")
        model.save(user_model_path)
        print("Saved model to", user_model_path)
    conn.close()

if __name__ == "__main__":
    main_train_all()
