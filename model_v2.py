# ai-backend/model_v2.py
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
import os

# --- helper parse ---
def to_date(s):
    if isinstance(s, str):
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    if isinstance(s, datetime):
        return s.date()
    return s

# -----------------------
# TRAIN / PREDICT utilities
# -----------------------

def aggregate_monthly(transactions: List[Dict[str,Any]]):
    # return DataFrame with index yyyy-mm, columns: income, expense, total, per-category
    rows = []
    for t in transactions:
        d = to_date(t['date'])
        rows.append({'ym': d.strftime("%Y-%m"), 'type': t['type'], 'amount': float(t['amount'] or 0), 'category': t.get('categoryId','unknown')})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df_income = df[df.type=='income'].groupby('ym').amount.sum().rename('income')
    df_exp = df[df.type!='income'].groupby('ym').amount.sum().rename('expense')
    df_total = pd.concat([df_income, df_exp], axis=1).fillna(0)
    df_total['net'] = df_total['income'] - df_total['expense']
    df_total = df_total.sort_index()
    return df_total

def build_features_for_monthly(df_monthly: pd.DataFrame, lags=3):
    # df_monthly index: yyyy-mm
    X, y = [], []
    idxs = df_monthly.index.tolist()
    for i in range(lags, len(idxs)):
        row = []
        # lag income and expense
        for lag in range(1, lags+1):
            row.append(df_monthly.iloc[i-lag].income)
            row.append(df_monthly.iloc[i-lag].expense)
        # moving averages
        row.append(df_monthly.iloc[i-lags:i].expense.mean())
        row.append(df_monthly.iloc[i-lags:i].income.mean())
        X.append(row)
        y.append(df_monthly.iloc[i].expense)
    return np.array(X), np.array(y)

# -----------------------
# Model container helpers
# -----------------------
MODEL_DIR = os.environ.get('AI_MODEL_DIR','./ai_models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'model_v2.joblib')

class V2Model:
    def __init__(self):
        # higher-level regressor for monthly expense
        self.monthly_regressor = None  # GradientBoostingRegressor
        # per-category regressors (dict cat -> regressor)
        self.cat_regressors = {}
        # metadata
        self.lags = 3

    def save(self, path=MODEL_PATH):
        joblib.dump({
            'monthly_regressor': self.monthly_regressor,
            'cat_regressors': self.cat_regressors,
            'lags': self.lags
        }, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            return cls()
        data = joblib.load(path)
        m = cls()
        m.monthly_regressor = data.get('monthly_regressor')
        m.cat_regressors = data.get('cat_regressors', {})
        m.lags = data.get('lags', 3)
        return m

# -----------------------
# Training function
# -----------------------
def train_model_from_transactions(transactions: List[Dict[str,Any]]):
    """
    Train:
      - monthly_regressor: predict next month's total expense
      - cat_regressors: for each top category, a simple LinearRegression on monthly expense per category
    """
    dfm = aggregate_monthly(transactions)
    if dfm.empty or len(dfm) < 4:
        # not enough data
        model = V2Model()
        return model

    # monthly regressor
    X, y = build_features_for_monthly(dfm, lags=3)
    if len(y) < 2:
        model = V2Model()
        return model

    reg = GradientBoostingRegressor(n_estimators=200, random_state=42)
    reg.fit(X, y)

    # per-category: aggregate per month per category
    rows = []
    for t in transactions:
        d = to_date(t['date'])
        ym = d.strftime("%Y-%m")
        rows.append({'ym': ym, 'category': t.get('categoryId','unknown'), 'type': t['type'], 'amount': float(t['amount'] or 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        cats = []
    else:
        df_exp = df[df.type!='income'].groupby(['ym','category']).amount.sum().reset_index()
        # pivot per category
        pivot = df_exp.pivot(index='ym', columns='category', values='amount').fillna(0)
        # pick top categories by recent total
        recent_sum = pivot.sum().sort_values(ascending=False)
        top_cats = recent_sum.head(10).index.tolist()
        cat_regs = {}
        for cat in top_cats:
            ser = pivot[cat]
            # if not enough non-zero months, skip linear fit, fallback to mean
            if ser.sum() == 0:
                continue
            # build simple lag features for cat
            s = ser.values
            if len(s) >= 4:
                Xc, yc = [], []
                lag = 3
                for i in range(lag, len(s)):
                    feat = []
                    for j in range(1, lag+1):
                        feat.append(s[i-j])
                    Xc.append(feat)
                    yc.append(s[i])
                lr = LinearRegression()
                lr.fit(np.array(Xc), np.array(yc))
                cat_regs[cat] = lr
            else:
                # fallback store mean
                cat_regs[cat] = ('mean', float(s.mean()))
    model = V2Model()
    model.monthly_regressor = reg
    model.cat_regressors = cat_regs
    model.save()
    return model

# -----------------------
# Serving: predict functions
# -----------------------
def predict_next_month_expense_from_transactions(model: V2Model, transactions: List[Dict[str,Any]]):
    dfm = aggregate_monthly(transactions)
    if dfm.empty:
        return 0.0, 'no_data'
    # if no trained regressor, fallback to simple heuristic (avg of last 3 months)
    if not model.monthly_regressor:
        last_n = dfm['expense'].tail(3).mean()
        return float(last_n), 'heuristic_avg'
    # build feature vector from last lags
    lags = model.lags
    if len(dfm) < lags:
        arr = np.concatenate([np.zeros((lags-len(dfm),2)), dfm[['income','expense']].values])
    else:
        arr = dfm[['income','expense']].values
    # take last lags rows
    last = arr[-lags:]
    feat = []
    for i in range(lags):
        feat.append(last[-1-i][0])  # income lag1..lagN (we reverse below)
        feat.append(last[-1-i][1])
    feat = feat[::-1]  # make order lag1,lag1_exp,lag2,...
    # append moving means
    feat.append(arr[-lags:,1].mean())
    feat.append(arr[-lags:,0].mean())
    X = np.array(feat).reshape(1,-1)
    pred = float(model.monthly_regressor.predict(X)[0])
    if pred < 0:
        pred = 0.0
    return pred, 'model_v2'

def predict_category_spend_next_month(model: V2Model, transactions: List[Dict[str,Any]]):
    # returns dict category -> predicted amount
    rows = []
    for t in transactions:
        d = to_date(t['date'])
        rows.append({'ym': d.strftime("%Y-%m"), 'category': t.get('categoryId','unknown'), 'type': t.get('type','expense'), 'amount': float(t.get('amount') or 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df_exp = df[df.type!='income'].groupby(['ym','category']).amount.sum().reset_index()
    pivot = df_exp.pivot(index='ym', columns='category', values='amount').fillna(0)
    preds = {}
    for cat, reg in model.cat_regressors.items():
        if isinstance(reg, tuple) and reg[0] == 'mean':
            preds[cat] = float(reg[1])
        else:
            s = pivot.get(cat)
            if s is None:
                # no data -> 0
                preds[cat] = 0.0
            else:
                arr = s.values
                # need last lag values
                lag = 3
                if len(arr) < lag:
                    feat = np.concatenate([np.zeros(lag-len(arr)), arr])[-lag:]
                else:
                    feat = arr[-lag:]
                feat = feat[::-1].reshape(1,-1)
                p = float(reg.predict(feat)[0])
                preds[cat] = max(0.0, p)
    # also include residual (other categories) as mean of others
    return preds

def predict_30d_balance_and_burn_date(transactions: List[Dict[str,Any]], current_balance: float, avg_daily_net_override: Optional[float]=None):
    """
    Compute avg daily net from transactions and project 30 days.
    avg_daily_net_override: if provided, use it instead of computed
    Return: projected_balance_30d, burn_date_or_none
    """
    if not transactions:
        return current_balance, None
    # group by date
    day_map = defaultdict(float)
    for t in transactions:
        d = to_date(t['date'])
        key = d.strftime("%Y-%m-%d")
        amt = float(t.get('amount') or 0)
        if t.get('type') == 'income':
            day_map[key] += amt
        else:
            day_map[key] -= amt
    # get recent 60 days
    days = sorted(day_map.keys())
    amounts = [day_map[d] for d in days]
    if avg_daily_net_override is not None:
        avg = avg_daily_net_override
    else:
        avg = float(np.mean(amounts)) if amounts else 0.0
    proj_30 = current_balance + avg * 30.0
    # estimate burn date: find day N where current_balance + avg*N <= 0
    burn_date = None
    if avg < 0:
        days_to_zero = int(np.ceil(-current_balance / avg)) if avg != 0 else None
        if days_to_zero is not None and days_to_zero >= 0:
            burn_date = (datetime.utcnow().date() + timedelta(days=days_to_zero)).isoformat()
    return float(proj_30), burn_date

# -----------------------
# Convenience: top-level predict wrapper
# -----------------------
def predict_all(model: V2Model, transactions: List[Dict[str,Any]], current_balance: Optional[float]=0.0):
    next_month_expense, method = predict_next_month_expense_from_transactions(model, transactions)
    per_cat = predict_category_spend_next_month(model, transactions)
    proj_30_balance, burn_date = predict_30d_balance_and_burn_date(transactions, float(current_balance or 0.0))
    return {
        'next_month_expense': next_month_expense,
        'next_month_method': method,
        'per_category_prediction': per_cat,
        'projected_balance_30d': proj_30_balance,
        'predicted_burn_date': burn_date
    }
