import os
import sqlite3
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, current_app, g, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from groq import Groq
# config
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)
RECO_MODEL_FILE = MODEL_PATH / "reco_model.pkl"
RECO_META_FILE = MODEL_PATH / "reco_meta.pkl"

bp = Blueprint("integrations", __name__, url_prefix="/api")
CORS_bp = None

def get_sqlite_conn():
    # Reuse same DB file used by app.py (library.db)
    db_path = os.getenv("LIB_DB_PATH", "library.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------
# Recommendation utilities
# -------------------------
def build_interaction_matrix():
    """
    Build a user x book interaction matrix from loans table.
    We consider each loan as a positive interaction (1).
    """
    conn = get_sqlite_conn()
    df_loans = pd.read_sql_query("SELECT id,book_id,member_id,borrowed_at,returned_at FROM loans", conn)
    conn.close()
    if df_loans.empty:
        return None, None, None

    # implicit feedback: count borrow occurrences per (member, book)
    df = df_loans.groupby(["member_id","book_id"]).size().reset_index(name="count")
    # pivot to matrix
    interactions = df.pivot(index="member_id", columns="book_id", values="count").fillna(0)
    return interactions, df["member_id"].unique(), df["book_id"].unique()

def train_recommender(n_components=50):
    """
    Train a simple SVD-based recommender (TruncatedSVD on interaction matrix).
    Save model to disk as RECO_MODEL_FILE and RECO_META_FILE.
    """
    interactions, members, books = build_interaction_matrix()
    if interactions is None:
        return {"status":"no_data"}

    # Use TruncatedSVD for latent factors
    X = interactions.values
    # if shape small, reduce components
    n_comp = min(n_components, min(X.shape)-1) if min(X.shape) > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    latent = svd.fit_transform(X)  # users -> latent
    # store model and metadata
    with open(RECO_MODEL_FILE, "wb") as f:
        pickle.dump({"svd": svd, "members_index": list(interactions.index), "books_index": list(interactions.columns)}, f)
    return {"status":"trained","members":len(interactions.index),"books":len(interactions.columns)}

def recommend_for_member(member_id, topk=8):
    """
    Load saved SVD model; compute nearest books by reconstructing scores.
    Returns list of book_ids recommended (descending score).
    """
    if not RECO_MODEL_FILE.exists():
        return {"status":"no_model"}
    with open(RECO_MODEL_FILE, "rb") as f:
        obj = pickle.load(f)
    svd = obj["svd"]
    members_index = list(obj["members_index"])
    books_index = list(obj["books_index"])

    # read interaction matrix again to align shapes
    interactions, _, _ = build_interaction_matrix()
    if interactions is None:
        return {"status":"no_data"}

    if member_id not in interactions.index:
        # cold start: recommend most popular books
        conn = get_sqlite_conn()
        df_pop = pd.read_sql_query("SELECT book_id, COUNT(*) as times FROM loans GROUP BY book_id ORDER BY times DESC LIMIT ?", conn, params=(topk,))
        conn.close()
        return {"status":"cold_start","recommendations": df_pop["book_id"].tolist()}

    user_vec = interactions.loc[member_id].values.reshape(1, -1)
    user_latent = svd.transform(user_vec)  # 1 x k
    # approximate scores: user_latent dot VT (svd.components_)
    approx_scores = np.dot(user_latent, svd.components_).flatten()
    # mask already-seen books
    seen = set(interactions.loc[member_id].loc[interactions.loc[member_id] > 0].index.tolist())
    # create book score pairs
    scores = list(zip(books_index, approx_scores))
    # filter and sort
    filtered = [ (b, s) for (b,s) in scores if b not in seen ]
    top = sorted(filtered, key=lambda x: x[1], reverse=True)[:topk]
    return {"status":"ok", "recommendations": [int(b) for b,_ in top]}

# -------------------------
# Analytics utilities
# -------------------------
def analytics_summary():
    """
    Return simple analytics: most borrowed books, late returns percentage, average borrow duration.
    This is a lightweight implementation suitable as an admin insight.
    """
    conn = get_sqlite_conn()
    df_loans = pd.read_sql_query("SELECT book_id, member_id, borrowed_at, due_at, returned_at FROM loans", conn)
    conn.close()
    if df_loans.empty:
        return {"status":"no_data"}

    # most borrowed
    most_borrowed = df_loans.groupby("book_id").size().sort_values(ascending=False).head(10).reset_index().rename(columns={0:"count"})
    most_borrowed_list = most_borrowed["book_id"].astype(int).tolist()

    # late returns
    def parse_dt(x):
        try:
            return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        except:
            return None

    df_loans["due_dt"] = df_loans["due_at"].apply(parse_dt)
    df_loans["returned_dt"] = df_loans["returned_at"].apply(parse_dt)
    df_loans["late"] = df_loans.apply(lambda r: (r["returned_dt"] is not None and r["returned_dt"] > r["due_dt"]) , axis=1)
    late_pct = 0.0
    if len(df_loans) > 0:
        late_pct = df_loans["late"].sum() / len(df_loans) * 100

    # avg borrow duration (for returned)
    df_returned = df_loans[df_loans["returned_dt"].notna()].copy()
    if not df_returned.empty:
        df_returned["duration_days"] = (df_returned["returned_dt"] - df_returned["due_dt"]).dt.days
        avg_duration = float(df_returned["duration_days"].abs().mean())
    else:
        avg_duration = None

    return {
        "status":"ok",
        "most_borrowed_book_ids": most_borrowed_list,
        "late_return_percentage": round(late_pct,2),
        "avg_return_delay_days": avg_duration
    }

# -------------------------
# Chat proxy (Groq or other)
# -------------------------
def chat_with_model(prompt, model="llama-3.1-8b-instant", max_tokens=512):
    """
    Minimal wrapper to call Groq. If you want to use other APIs (OpenAI), update here.
    Note: The groq client must be installed and a valid key used when registering integrations.
    """
    # client was stored on blueprint during register_extensions
    client = current_app.config.get("AI_CLIENT")
    if client is None:
        raise RuntimeError("AI client not configured")
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful library assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return chat_completion.choices[0].message.content

# -------------------------
# Blueprint routes
# -------------------------
@bp.route("/train_recommender", methods=["POST"])
def route_train_recommender():
    res = train_recommender()
    return jsonify(res)

@bp.route("/recommendations/<int:member_id>", methods=["GET"])
def route_recommend(member_id):
    topk = int(request.args.get("topk", 8))
    res = recommend_for_member(member_id, topk=topk)
    # expand book ids to book metadata
    if res.get("status") == "ok" or res.get("status") == "cold_start":
        conn = get_sqlite_conn()
        book_ids = res.get("recommendations", [])
        if book_ids:
            q = ",".join(["?"]*len(book_ids))
            rows = conn.execute(f"SELECT id,title,author,isbn FROM books WHERE id IN ({q})", book_ids).fetchall()
            conn.close()
            books = [dict(r) for r in rows]
            res["books"] = books
    return jsonify(res)

@bp.route("/analytics", methods=["GET"])
def route_analytics():
    res = analytics_summary()
    return jsonify(res)

@bp.route("/chat", methods=["POST"])
def route_chat():
    data = request.json or {}
    prompt = data.get("prompt") or data.get("question") or ""
    model = data.get("model", current_app.config.get("AI_DEFAULT_MODEL","llama-3.1-8b-instant"))
    try:
        out = chat_with_model(prompt, model=model)
        return jsonify({"status":"ok","response": out})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 500

# -------------------------
# Register helper
# -------------------------
def register_extensions(app, groq_api_key=None):
    """
    Call this from your app.py:
      import integrations
      integrations.register_extensions(app, groq_api_key="gsk_...")
    Or call without key and set env var GROQ_API_KEY before starting.
    """
    global CORS_bp
    # 1) attach blueprint
    app.register_blueprint(bp)
    # enable CORS for blueprint
    CORS_bp = CORS(app, resources={r"/api/*": {"origins": "*"}})

    # 2) configure Groq client
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            client = Groq(api_key=api_key)
            app.config["AI_CLIENT"] = client
            app.config["AI_DEFAULT_MODEL"] = os.getenv("AI_DEFAULT_MODEL","llama-3.1-8b-instant")
        except Exception as e:
            app.config["AI_CLIENT"] = None
            print("Warning: failed to init Groq client:", e)
    else:
        print("Warning: no GROQ_API_KEY provided — AI endpoints will fail until client is configured.")
    print("Integrations registered: /api/recommendations, /api/analytics, /api/chat, /api/train_recommender")
