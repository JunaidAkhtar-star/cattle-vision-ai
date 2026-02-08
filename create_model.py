"""
create_model.py

Utility script to create the SQLite database and all tables
(User, Livestock, Prediction) for the CATTLE BREED 3 project.

Run:
    (venv) python create_model.py
"""

from datetime import datetime
import os

from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# Minimal Flask + DB setup (same config as app.py)
# -------------------------------------------------------------------
app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "sqlite:///instance/cattle_vision.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")

# Ensure instance directory exists for SQLite file
os.makedirs("instance", exist_ok=True)

db = SQLAlchemy(app)

# -------------------------------------------------------------------
# Models (must match exactly with app.py)
# -------------------------------------------------------------------

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    role = db.Column(db.String(50))        # Farmer / FLW / Veterinarian
    location = db.Column(db.String(200))
    password_hash = db.Column(db.String(255))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Livestock(db.Model):
    __tablename__ = "livestock"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    animal_id = db.Column(db.String(100), nullable=False)  # tag / ear number
    species = db.Column(db.String(50))                     # Cattle / Buffalo
    breed = db.Column(db.String(100))
    age_years = db.Column(db.Float)
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    livestock_id = db.Column(db.Integer, db.ForeignKey("livestock.id"), nullable=False)

    image_path = db.Column(db.String(300), nullable=False)

    predicted_breed = db.Column(db.String(100))
    breed_confidence = db.Column(db.Float)

    predicted_disease = db.Column(db.String(100))
    disease_confidence = db.Column(db.Float)

    health_status = db.Column(db.String(100))
    notes = db.Column(db.Text)

    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)

# -------------------------------------------------------------------
# Main: create all tables
# -------------------------------------------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("Database and tables created successfully at instance/cattle_vision.db")
