import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "cattle_vision.db")

print("Using DB:", DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Add columns only if they do not exist yet
def add_column(stmt):
    try:
        cur.execute(stmt)
        print("Executed:", stmt)
    except sqlite3.OperationalError as e:
        print("Skip (maybe already exists):", stmt, "| Error:", e)

add_column("ALTER TABLE users ADD COLUMN is_email_verified INTEGER DEFAULT 0;")
add_column("ALTER TABLE users ADD COLUMN email_verified_at TEXT;")
add_column("ALTER TABLE users ADD COLUMN otp_code TEXT;")
add_column("ALTER TABLE users ADD COLUMN otp_expires_at TEXT;")

conn.commit()
conn.close()

print("Migration done.")
