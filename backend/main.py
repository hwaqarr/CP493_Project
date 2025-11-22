#Hashim Waqar
#Cp493
#Main.py

import os
import webbrowser
import sqlite3

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from groq import Groq


try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from backend.database import init_db, insert_steps, get_all_steps, insert_prediction

print("Using database at:", os.path.abspath("steps.db"))

app = FastAPI()
init_db()


@app.get("/")
def home():
    return {"message": "FastAPI and SQLite system are running."}


@app.post("/steps")
def add_steps(user_id: str, date: str, steps: int):
    insert_steps(user_id, date, steps)
    return {"message": "Steps added", "user_id": user_id, "date": date, "steps": steps}


@app.get("/steps/{user_id}")
def read_steps(user_id: str):
    rows = get_all_steps(user_id)
    data = [{"date": r[0], "steps": r[1]} for r in rows]
    return {"user_id": user_id, "steps": data}


@app.get("/insights/{user_id}")
def generate_insights(user_id: str):
    # --- read data ---
    conn = sqlite3.connect("steps.db")
    try:
        df = pd.read_sql_query(
            "SELECT date, steps FROM steps WHERE user_id = ?",
            conn,
            params=(user_id,),
        )
    finally:
        conn.close()

    if df.empty:
        return {"insights": ["No data available for this user."]}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- last 7 recorded days vs previous 7 ---
    recent_week = df.tail(7)
    prev_week = df.iloc[-14:-7] if len(df) >= 14 else pd.DataFrame()

    avg_recent = float(recent_week["steps"].mean()) if not recent_week.empty else 0.0
    avg_prev = float(prev_week["steps"].mean()) if not prev_week.empty else 0.0
    change = ((avg_recent - avg_prev) / avg_prev * 100.0) if avg_prev > 0 else 0.0

    try:
        top_day = df.groupby(df["date"].dt.day_name())["steps"].mean().idxmax()
    except ValueError:
        top_day = "N/A"

    last3_avg = float(df.tail(3)["steps"].mean()) if len(df) >= 1 else 0.0

    baseline_summary = (
        f"In the past week, your average was {int(avg_recent):,} steps. "
        f"That’s {abs(change):.1f}% "
        f"{'higher' if change > 0 else 'lower' if change < 0 else 'the same'} than the previous week. "
        f"Your most active day is {top_day}, and your last three-day average is {int(last3_avg):,} steps."
    )

    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not groq_api_key:
        return {"insights": [baseline_summary]}

    # Allow override via env, then try a few known-good models.
    model_candidates = []
    env_model = (os.getenv("GROQ_MODEL") or "").strip()
    if env_model:
        model_candidates.append(env_model)

    model_candidates += [
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mistral-saba-24b",
        "llama-3.3-70b-versatile",
    ]

    client = Groq(api_key=groq_api_key)

    for model_id in model_candidates:
        try:
            resp = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0.4,  # a bit steadier
                max_completion_tokens=160,
                presence_penalty=0.1,
                frequency_penalty=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write like a motivational coach. Keep it concise and specific. "
                            "Use natural phrasing, no jargon, no emojis, no bullet points, no AI disclaimers."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Turn this into a short, encouraging summary in 4–5 sentences. "
                            "Be concrete and natural:\n\n" + baseline_summary
                        ),
                    },
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return {"insights": [text]}
        except Exception as e:
            print(f"[GROQ ERROR] {model_id}: {e}")

    # Fallback to baseline if every model fails
    return {"insights": [baseline_summary]}


@app.get("/predict/{user_id}")
def predict_steps(user_id: str):
    conn = sqlite3.connect("steps.db")
    try:
        df = pd.read_sql_query(
            "SELECT date, steps FROM steps WHERE user_id = ?", conn, params=(user_id,)
        )
    finally:
        conn.close()

    if df.empty or len(df) < 3:
        return {"prediction": "Not enough data to make a prediction."}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["day_num"] = range(len(df))

    model = LinearRegression()
    model.fit(df[["day_num"]], df["steps"])

    # Predict with the SAME named column to avoid the sklearn warning
    next_day = pd.DataFrame({"day_num": [df["day_num"].max() + 1]})
    predicted_steps = int(model.predict(next_day)[0])

    return {"prediction": f"Predicted steps for tomorrow: {predicted_steps:,}"}


class StepData(BaseModel):
    user_id: str
    date: str
    steps: int


# NEW: model for chat requests
class ChatMessage(BaseModel):
    user_id: str
    history: list[list[str]]  # [[user, assistant], ...]
    message: str


@app.post("/predict")
def predict_from_n8n(data: StepData):
    insert_steps(data.user_id, data.date, data.steps)

    conn = sqlite3.connect("steps.db")
    try:
        df = pd.read_sql_query(
            "SELECT date, steps FROM steps WHERE user_id = ?", conn, params=(data.user_id,)
        )
    finally:
        conn.close()

    if df.empty or len(df) < 3:
        return {"prediction": "Not enough data yet for prediction."}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["day_num"] = range(len(df))

    model = LinearRegression()
    model.fit(df[["day_num"]], df["steps"])

    next_day = pd.DataFrame({"day_num": [df["day_num"].max() + 1]})
    predicted_steps = int(model.predict(next_day)[0])

    insert_prediction(data.user_id, data.date, data.steps, predicted_steps)

    return {
        "user_id": data.user_id,
        "today_steps": data.steps,
        "predicted_tomorrow": predicted_steps,
    }


@app.get("/predictions/{user_id}")
def get_predictions(user_id: str):
    conn = sqlite3.connect("steps.db")
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT date, steps, predicted_steps FROM predictions WHERE user_id = ? ORDER BY date ASC",
            (user_id,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"predictions": []}

    return {
        "predictions": [
            {"date": r[0], "steps": r[1], "prediction": r[2]} for r in rows
        ]
    }


# NEW: conversational chat endpoint
@app.post("/chat")
def chat_with_ai(payload: ChatMessage):
    user_id = payload.user_id

    # --- read data for context ---
    conn = sqlite3.connect("steps.db")
    try:
        df = pd.read_sql_query(
            "SELECT date, steps FROM steps WHERE user_id = ?",
            conn,
            params=(user_id,),
        )
    finally:
        conn.close()

    if df.empty:
        baseline_summary = "This user has no recorded step data yet."
    else:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        recent_week = df.tail(7)
        prev_week = df.iloc[-14:-7] if len(df) >= 14 else pd.DataFrame()

        avg_recent = float(recent_week["steps"].mean()) if not recent_week.empty else 0.0
        avg_prev = float(prev_week["steps"].mean()) if not prev_week.empty else 0.0
        change = ((avg_recent - avg_prev) / avg_prev * 100.0) if avg_prev > 0 else 0.0

        try:
            top_day = df.groupby(df["date"].dt.day_name())["steps"].mean().idxmax()
        except ValueError:
            top_day = "N/A"

        last3_avg = float(df.tail(3)["steps"].mean()) if len(df) >= 1 else 0.0

        baseline_summary = (
            f"In the past week, the user's average was {int(avg_recent):,} steps. "
            f"That’s {abs(change):.1f}% "
            f"{'higher' if change > 0 else 'lower' if change < 0 else 'the same'} than the previous week. "
            f"Their most active day is {top_day}, and their last three-day average is {int(last3_avg):,} steps."
        )

    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not groq_api_key:
        return {
            "reply": (
                "AI chat is not configured yet because GROQ_API_KEY is "
                "missing on the server."
            )
        }

    # same model selection pattern as /insights
    model_candidates = []
    env_model = (os.getenv("GROQ_MODEL") or "").strip()
    if env_model:
        model_candidates.append(env_model)

    model_candidates += [
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mistral-saba-24b",
        "llama-3.3-70b-versatile",
    ]

    client = Groq(api_key=groq_api_key)

    # build chat messages with history
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly, motivational walking and activity coach. "
                "You chat with the user about their steps, goals, and habits. "
                "Keep answers concise, practical, and encouraging."
            ),
        },
        {
            "role": "system",
            "content": "Context about this user's recent activity: " + baseline_summary,
        },
    ]

    for turn in payload.history:
        if len(turn) != 2:
            continue
        user_text, assistant_text = turn
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": payload.message})

    for model_id in model_candidates:
        try:
            resp = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0.6,
                max_completion_tokens=256,
                presence_penalty=0.2,
                frequency_penalty=0.2,
                messages=messages,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return {"reply": text}
        except Exception as e:
            print(f"[GROQ CHAT ERROR] {model_id}: {e}")

    return {
        "reply": (
            "I couldn't generate a reply right now, but your data and "
            "insights are still available."
        )
    }







