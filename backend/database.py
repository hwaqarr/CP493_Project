#Hashim Waqar
#Cp493
#Database.py


import sqlite3

import os
DB_NAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "steps.db")



def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            steps INTEGER NOT NULL,
            UNIQUE(user_id, date)
        )
    """)
    conn.commit( )
    conn.close()


def insert_steps(user_id: str, date: str, steps: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO steps (user_id, date, steps)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, date) DO UPDATE SET steps = excluded.steps
    """, (user_id, date, steps))
    conn.commit()
    conn.close()


def get_all_steps(user_id: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date, steps FROM steps WHERE user_id = ? ORDER BY date ASC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def insert_prediction(user_id: str, date: str, steps: int, predicted_steps: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            steps INTEGER NOT NULL,
            predicted_steps INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        INSERT INTO predictions (user_id, date, steps, predicted_steps)
        VALUES (?, ?, ?, ?)
    """, (user_id, date, steps, predicted_steps))
    conn.commit()
    conn.close()
