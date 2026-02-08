from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash,
)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
import os
import math
import socket
from dotenv import load_dotenv
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import smtplib
import ssl
import random
import requests
import json

# Local imports for models/predictors
from src.combined_predictor import CombinedPredictor
from src.predict_disease import DiseasePredictor
from src.config_disease import DISEASE_MODEL_PATH, IMG_SIZE

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# Initialize Flask app
# ---------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------
# Configuration (absolute SQLite path to avoid errors)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "cattle_vision.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Google OAuth config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "http://127.0.0.1:5000/login/google/callback",
)
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Initialize DB
db = SQLAlchemy(app)

# ---------------------------------------------------
# Load India states and districts JSON (ALL states)
# ---------------------------------------------------
INDIA_STATE_DISTRICTS = {}
data_path = os.path.join(BASE_DIR, "data", "india_states_districts.json")
if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        INDIA_STATE_DISTRICTS = json.load(f)

print("JSON path:", data_path)
print("States loaded:", len(INDIA_STATE_DISTRICTS))

# ---------------------------------------------------
# Database models
# ---------------------------------------------------
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    role = db.Column(db.String(50))  # Farmer / FLW / Veterinarian
    location = db.Column(db.String(200))
    password_hash = db.Column(db.String(255))

    # farm location for nearest vet feature (not used in new flow, but kept)
    farm_lat = db.Column(db.Float)   # optional: user's farm latitude
    farm_lng = db.Column(db.Float)   # optional: user's farm longitude

    is_email_verified = db.Column(db.Boolean, default=False)
    email_verified_at = db.Column(db.DateTime)
    otp_code = db.Column(db.String(6))
    otp_expires_at = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    livestock = db.relationship("Livestock", backref="owner", lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)


class Livestock(db.Model):
    __tablename__ = "livestock"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    animal_id = db.Column(db.String(100), nullable=False)
    species = db.Column(db.String(50))  # Cattle / Buffalo
    breed = db.Column(db.String(100))
    age_years = db.Column(db.Float)
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship("Prediction", backref="livestock", lazy=True)


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


class VetCenter(db.Model):
    __tablename__ = "vet_centers"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lng = db.Column(db.Float, nullable=False)
    phone = db.Column(db.String(30))
    opening_hours = db.Column(db.String(120))
    state = db.Column(db.String(80))
    district = db.Column(db.String(80))
    center_type = db.Column(db.String(80))  # Govt / Private / NGO


# ------------ NEW: Feedback model ------------
class PredictionFeedback(db.Model):
    __tablename__ = "prediction_feedback"

    id = db.Column(db.Integer, primary_key=True)

    image_id = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    predicted_breed = db.Column(db.String(100))
    predicted_disease = db.Column(db.String(100))
    correct_breed = db.Column(db.String(100))
    correct_disease = db.Column(db.String(100))

    is_correct = db.Column(db.Boolean, default=None)
    comment = db.Column(db.Text)

    prediction_confidence = db.Column(db.Float)
    model_version = db.Column(db.String(32))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------------------------------------------
# Breed configuration and model
# ---------------------------------------------------
breed_names = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur", "Bhadawari",
    "Brown Swiss", "Dangi", "Deoni", "Gir", "Guernsey", "Hallikar",
    "Holstein Fresian", "Red Dane",
]

buffalo_list = ["Bhadawari", "Banni"]

breed_traits = {
    "Alambadi": "Draught type, native to Tamil Nadu.",
    "Amritmahal": "Draught type, good for ploughing.",
    "Ayrshire": "Dairy breed, Scottish origin.",
    "Banni": "Buffalo, hardy and disease resistant.",
    "Bargur": "Draught breed, agile and active.",
    "Bhadawari": "Buffalo, high milk fat.",
    "Brown Swiss": "Dairy, high yield and gentle.",
    "Dangi": "Dual purpose, resistant to harsh conditions.",
    "Deoni": "Dual purpose, good milk and work.",
    "Gir": "Dairy, high milk yield, heat tolerant.",
    "Guernsey": "Dairy, golden milk.",
    "Hallikar": "Draught, used for ploughing.",
    "Holstein Fresian": "High-yield dairy breed, black and white coat.",
    "Red Dane": "Danish dairy breed, good milk production.",
}

# --- richer breed explanations for output ---
BREED_EXPLANATIONS = {
    "Alambadi": "Draught-type cattle from Tamil Nadu with strong build used for ploughing.",
    "Amritmahal": "Powerful draught breed known for endurance in field work.",
    "Ayrshire": "Medium-sized dairy breed with red and white markings and good milk yield.",
    "Banni": "Buffalo breed adapted to arid Kutch region, hardy and disease resistant.",
    "Bargur": "Active draught cattle with compact body and agile movement, used in hilly areas.",
    "Bhadawari": "Buffalo with high-fat milk, copper-colored body and strong horns.",
    "Brown Swiss": "Large dairy cattle with brown coat and calm temperament.",
    "Dangi": "Medium-sized dual-purpose cattle adapted to heavy rainfall and hilly terrain.",
    "Deoni": "Dual-purpose breed giving good milk and capable of farm work.",
    "Gir": "Distinctive dairy breed with domed forehead, long ears and high milk yield.",
    "Guernsey": "Dairy cattle producing golden-colored milk rich in butterfat.",
    "Hallikar": "Strong draught breed used traditionally for ploughing and transport.",
    "Holstein Fresian": "Large black and white dairy breed known for very high milk yield.",
    "Red Dane": "Red-coated Danish dairy breed known for good milk production.",
}

def load_breed_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(breed_names))
    state_dict = torch.load("models/best_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

breed_model = load_breed_model()

breed_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

def predict_breed(image_path: str):
    image = Image.open(image_path).convert("RGB")
    input_tensor = breed_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = breed_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# ---------------------------------------------------
# Disease predictor and combined predictor
# ---------------------------------------------------
disease_predictor = DiseasePredictor(
    model_path=DISEASE_MODEL_PATH,
    image_size=IMG_SIZE,
)

combined_predictor = CombinedPredictor(
    breed_predictor=predict_breed,
    disease_predictor=disease_predictor,
    breed_names=breed_names,
)

# -------- disease risk helpers for explanation --------
def disease_risk_band(score: float):
    if score < 0.3:
        return "low"
    elif score < 0.7:
        return "medium"
    else:
        return "high"

def disease_explanation(score: float, disease_name: str):
    band = disease_risk_band(score)
    if band == "low":
        return f"No strong visual signs of {disease_name} detected."
    elif band == "medium":
        return f"Some visible skin or body changes that may be early signs of {disease_name}."
    else:
        return f"Marked visible changes strongly suggest {disease_name}. Please confirm with a veterinarian."

def disease_recommendation(band: str):
    if band == "low":
        return {
            "urgency": "Monitor only",
            "steps": [
                "Keep observing this animal for any new lumps, wounds or fever.",
                "If the condition worsens, contact a veterinarian.",
            ],
        }
    elif band == "medium":
        return {
            "urgency": "Within 24–48 hours",
            "steps": [
                "Avoid mixing this animal with new animals.",
                "Contact a veterinarian for inspection within 1–2 days.",
            ],
        }
    else:
        return {
            "urgency": "Immediately",
            "steps": [
                "Isolate this animal from the herd.",
                "Contact a veterinarian or government helpline immediately.",
                "Do not move or sell this animal until cleared.",
            ],
        }

# ---------------------------------------------------
# Helper functions (session-based auth + email + geo)
# ---------------------------------------------------
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return User.query.get(user_id)

def get_or_create_livestock(user, species: str, breed: str):
    livestock = (
        Livestock.query.filter_by(user_id=user.id, species=species, breed=breed)
        .order_by(Livestock.id.asc())
        .first()
    )
    if livestock:
        return livestock

    livestock = Livestock(
        user_id=user.id,
        animal_id=f"{breed}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        species=species,
        breed=breed,
        age_years=None,
        notes="Auto-created animal record",
    )
    db.session.add(livestock)
    db.session.commit()
    return livestock

def generate_otp():
    return f"{random.randint(100000, 999999)}"

def send_email_otp(to_email, otp):
    smtp_server = "smtp.gmail.com"
    port = 465
    sender_email = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_PASSWORD")

    print("DEBUG SMTP:", sender_email, bool(password))

    if not sender_email or not password:
        print("SMTP credentials not configured")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Cattle Vision · Email Verification Code"
    msg["From"] = sender_email
    msg["To"] = to_email

    text = f"""Cattle Vision Email Verification

Your verification code is: {otp}
It will expire in 10 minutes.

If you did not request this, you can ignore this email.
"""

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f3faf7; padding: 0; margin: 0;">
        <table width="100%" cellpadding="0" cellspacing="0" style="padding: 24px 0;">
          <tr>
            <td align="center">
              <table width="520" cellpadding="0" cellspacing="0"
                     style="background-color: #ffffff; border-radius: 12px; padding: 24px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.06);">

                <tr>
                  <td align="center" style="padding-bottom: 12px;">
                    <div style="
                      height: 44px;
                      width: 44px;
                      border-radius: 12px;
                      background-color: #fbbf24;
                      display: flex;
                      align-items: center;
                      justify-content: center;
                      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1),
                                  0 4px 6px -4px rgba(0,0,0,0.1);
                      ">
                      <span style="
                        font-size: 20px;
                        font-weight: 700;
                        color: #047857;
                        ">
                        🐄
                      </span>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td align="center" style="padding-bottom: 10px;">
                    <div style="font-weight:700; font-size:18px; color:#064e3b;">
                      Cattle Vision AI
                    </div>
                  </td>
                </tr>

                <tr>
                  <td style="padding-bottom: 8px;">
                    <h2 style="margin: 0; color: #022c22; font-size: 22px;">
                      Email verification required
                    </h2>
                  </td>
                </tr>

                <tr>
                  <td style="padding-bottom: 12px; color:#064e3b; font-size:14px;">
                    Thank you for registering with <strong>Cattle Vision</strong>.
                    Use the verification code below to activate your account.
                  </td>
                </tr>

                <tr>
                  <td align="center" style="padding: 16px 0 20px 0;">
                    <div style="display:inline-block; padding: 14px 28px; border-radius: 12px;
                                background-color:#ecfdf5; border: 1px dashed #16a34a;
                                font-size: 26px; letter-spacing: 4px; color:#065f46; font-weight:700;">
                      {otp}
                    </div>
                  </td>
                </tr>

                <tr>
                  <td style="padding-bottom: 8px; color:#4b5563; font-size:13px;">
                    This code will expire in <strong>10 minutes</strong>. For your security,
                    do not share this code with anyone.
                  </td>
                </tr>

                <tr>
                  <td style="padding-top: 16px; border-top: 1px solid #e5e7eb;
                             color:#6b7280; font-size:12px;">
                    If you did not try to sign up to Cattle Vision, you can safely ignore this email.
                  </td>
                </tr>
                <tr>
                  <td style="padding-top: 8px; color:#9ca3af; font-size:11px;">
                    © 2025 Digital Dominators · Cattle Vision AI Project
                  </td>
                </tr>
              </table>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    msg.attach(part1)
    msg.attach(part2)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, msg.as_string())
        print("OTP email sent to:", to_email)
    except Exception as e:
        print("Error sending OTP email:", e)

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distance in kilometers between two lat/lng points.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route("/", methods=["GET", "POST"])
def upload():
    user = get_current_user()

    if request.method == "POST":
        if not user:
            flash("Please login or register before uploading.")
            return redirect(url_for("login"))

        if "file" not in request.files:
            flash("No file part in request.")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result_raw = combined_predictor.predict_all(filepath)

            # Breed probabilities and top-3
            breed_probs = list(zip(breed_names, result_raw["breed_probabilities"]))
            top_breed = result_raw["breed"]
            top_breed_conf = float(result_raw["breed_confidence"])

            # Build top-3 structure for UI
            probs_only = result_raw["breed_probabilities"]
            top3_idx = sorted(
                range(len(probs_only)),
                key=lambda i: probs_only[i],
                reverse=True,
            )[:3]
            top3_list = [
                {
                    "name": breed_names[i],
                    "confidence": float(probs_only[i]),
                }
                for i in top3_idx
            ]

            species = "Buffalo" if top_breed in buffalo_list else "Cattle"

            health_status = result_raw["health_status"]
            disease_name = (
                result_raw["disease"]
                if result_raw["disease"] != "Healthy"
                else "No detection"
            )
            disease_conf = (
                float(result_raw["disease_confidence"])
                if disease_name != "No detection"
                else 0.0
            )

            livestock = get_or_create_livestock(user, species=species, breed=top_breed)

            prediction = Prediction(
                livestock_id=livestock.id,
                image_path=filepath,
                predicted_breed=top_breed,
                breed_confidence=top_breed_conf,
                predicted_disease=None if disease_name == "No detection" else disease_name,
                disease_confidence=None if disease_name == "No detection" else disease_conf,
                health_status=health_status,
                notes=None,
                analysis_date=datetime.utcnow(),
            )
            db.session.add(prediction)
            db.session.commit()

            # --- structured result for UI/API ---
            disease_block = None
            recommendation_block = None
            if disease_name != "No detection":
                band = disease_risk_band(disease_conf)
                disease_block = {
                    "name": disease_name,
                    "risk_level": band,
                    "score": disease_conf,
                    "explanation": disease_explanation(disease_conf, disease_name),
                    "disclaimer": "This is an AI-based visual screening, not a clinical diagnosis.",
                }
                recommendation_block = disease_recommendation(band)
            else:
                disease_block = {
                    "name": "No obvious disease",
                    "risk_level": "low",
                    "score": 0.0,
                    "explanation": "No obvious visual signs of disease detected in this image.",
                    "disclaimer": "This is an AI-based visual screening, not a clinical diagnosis.",
                }
                recommendation_block = disease_recommendation("low")

            # simple image_id for feedback: use filename or DB id
            image_id = os.path.basename(filepath)
            model_version = "breed_v1"

            result_structured = {
                "breed": {
                    "top1": {
                        "name": top_breed,
                        "confidence": top_breed_conf,
                    },
                    "top3": top3_list,
                    "explanation": BREED_EXPLANATIONS.get(
                        top_breed,
                        "Typical body shape and coat pattern for this breed.",
                    ),
                },
                "disease": disease_block,
                "health_status": health_status,
                "recommendation": recommendation_block,
            }

            return render_template(
                "result.html",
                img_file=os.path.basename(filepath),
                result=result_structured,
                # legacy context
                breed_probs=breed_probs,
                top_breed=top_breed,
                top_breed_percentage=top_breed_conf,
                buffalo_list=buffalo_list,
                breed_traits=breed_traits,
                health_status=health_status,
                disease=disease_name,
                disease_confidence=disease_conf,
                user=user,
                livestock=livestock,
                prediction=prediction,
                # NEW for feedback
                image_id=image_id,
                confidence=top_breed_conf,
                model_version=model_version,
                # NEW: pass all breed names for feedback dropdown
                breed_options=breed_names,
            )

    recent_predictions = []
    if user:
        recent_predictions = (
            Prediction.query.join(Livestock)
            .filter(Livestock.user_id == user.id)
            .order_by(Prediction.analysis_date.desc())
            .limit(5)
            .all()
        )

    return render_template(
        "upload.html",
        user=user,
        recent_predictions=recent_predictions,
    )


@app.route("/history")
def history():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    livestock_list = Livestock.query.filter_by(user_id=user.id).all()

    return render_template(
        "history.html",
        user=user,
        livestock_list=livestock_list,
    )


@app.route("/livestock/add", methods=["GET", "POST"])
def add_livestock():
    user = get_current_user()
    if not user:
        flash("Please login to add livestock.")
        return redirect(url_for("login"))

    if request.method == "POST":
        animal_id = request.form.get("animal_id", "").strip()
        species = request.form.get("species", "").strip()
        breed = request.form.get("breed", "").strip()
        age_years = request.form.get("age_years") or None
        notes = request.form.get("notes", "").strip()

        if not animal_id:
            flash("Animal ID is required.")
            return render_template("add_livestock.html", user=user)

        livestock = Livestock(
            user_id=user.id,
            animal_id=animal_id,
            species=species,
            breed=breed,
            age_years=float(age_years) if age_years else None,
            notes=notes,
        )
        db.session.add(livestock)
        db.session.commit()
        flash("Livestock added successfully.")
        return redirect(url_for("history"))

    return render_template("add_livestock.html", user=user)


@app.route("/livestock/<int:livestock_id>")
def livestock_detail(livestock_id):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    animal = Livestock.query.filter_by(id=livestock_id, user_id=user.id).first()
    if not animal:
        flash("Livestock not found.")
        return redirect(url_for("history"))

    predictions = (
        Prediction.query
        .filter_by(livestock_id=animal.id)
        .order_by(Prediction.analysis_date.desc())
        .all()
    )

    return render_template(
        "livestock_detail.html",
        user=user,
        animal=animal,
        predictions=predictions,
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("upload"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        role = request.form.get("role", "Farmer")
        location = request.form.get("location", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Name, email and password are required.")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.")
            return redirect(url_for("login"))

        user = User(
            name=name,
            email=email,
            phone=phone,
            role=role,
            location=location,
        )
        user.set_password(password)

        otp = generate_otp()
        user.otp_code = otp
        user.otp_expires_at = datetime.utcnow() + timedelta(minutes=10)
        user.is_email_verified = False

        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash("Could not create user. Try different email.")
            return render_template("register.html")

        print("Sending OTP to:", user.email, "code:", otp)
        send_email_otp(user.email, otp)

        session["pending_user_id"] = user.id
        flash("We sent a verification code to your email. Enter it to activate your account.")
        return redirect(url_for("verify_email_otp"))

    return render_template("register.html")


@app.route("/verify-email", methods=["GET", "POST"])
def verify_email_otp():
    user_id = session.get("pending_user_id")
    if not user_id:
        flash("No verification in progress.")
        return redirect(url_for("register"))

    user = User.query.get(user_id)
    if not user:
        flash("User not found.")
        return redirect(url_for("register"))

    if request.method == "POST":
        code = request.form.get("otp", "").strip()
        now = datetime.utcnow()

        if not user.otp_code or not user.otp_expires_at or now > user.otp_expires_at:
            flash("OTP expired, please register again.")
            return redirect(url_for("register"))

        if code != user.otp_code:
            flash("Invalid OTP code.")
            return render_template("verify_email.html")

        user.is_email_verified = True
        user.email_verified_at = now
        user.otp_code = None
        user.otp_expires_at = None
        db.session.commit()

        session.pop("pending_user_id", None)
        session["user_id"] = user.id
        flash("Email verified successfully. You are now logged in.")
        return redirect(url_for("upload"))

    return render_template("verify_email.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid email or password.")
            return render_template("login.html")

        if not user.is_email_verified:
            flash("Please verify your email before logging in.")
            session["pending_user_id"] = user.id
            return redirect(url_for("verify_email_otp"))

        session["user_id"] = user.id
        flash("Logged in successfully.")
        return redirect(url_for("upload"))

    return render_template("login.html")


# ---------------- Google OAuth routes (login only) ----------------
@app.route("/login/google")
def login_with_google():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        flash("Google login is not configured.")
        return redirect(url_for("login"))

    from urllib.parse import urlencode

    if "devtunnels" in request.host:
        redirect_uri = "http://127.0.0.1:5000/login/google/callback"
    else:
        redirect_uri = request.host_url.rstrip("/") + url_for("google_callback")

    print("DEBUG login redirect_uri:", redirect_uri)

    state = os.urandom(16).hex()
    session["oauth_state"] = state
    print("DEBUG login set oauth_state:", state)

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "consent",
    }

    auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return redirect(auth_url)


@app.route("/login/google/callback")
def google_callback():
    state = request.args.get("state")
    stored_state = session.get("oauth_state")
    print("DEBUG callback state from Google:", state)
    print("DEBUG callback state in session:", stored_state)

    if not state or state != stored_state:
        flash("Invalid Google login state. Please try again.")
        return redirect(url_for("login"))

    code = request.args.get("code")
    if not code:
        flash("No code returned from Google.")
        return redirect(url_for("login"))

    if "devtunnels" in request.host:
        redirect_uri = "https://sx5b9f05-5000.inc1.devtunnels.ms/login/google/callback"
    else:
        redirect_uri = request.host_url.rstrip("/") + url_for("google_callback")

    print("DEBUG callback redirect_uri:", redirect_uri)

    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    token_resp = requests.post(GOOGLE_TOKEN_URL, data=data)
    if token_resp.status_code != 200:
        print("Token response:", token_resp.text)
        flash("Failed to get token from Google.")
        return redirect(url_for("login"))

    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        flash("No access token from Google.")
        return redirect(url_for("login"))

    headers = {"Authorization": f"Bearer {access_token}"}
    userinfo_resp = requests.get(GOOGLE_USERINFO_URL, headers=headers)
    if userinfo_resp.status_code != 200:
        print("Userinfo response:", userinfo_resp.text)
        flash("Failed to get user info from Google.")
        return redirect(url_for("login"))

    userinfo = userinfo_resp.json()
    email = userinfo.get("email", "").lower()
    name = userinfo.get("name") or email.split("@")[0]

    if not email:
        flash("Google account has no email.")
        return redirect(url_for("login"))

    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(
            name=name,
            email=email,
            phone="",
            role="Farmer",
            location="",
        )
        user.is_email_verified = True
        user.email_verified_at = datetime.utcnow()
        user.set_password(os.urandom(16).hex())
        db.session.add(user)
        db.session.commit()

    session["user_id"] = user.id
    flash("Logged in with Google successfully.")
    return redirect(url_for("upload"))


# ---------- Help request and nearest vets ----------
@app.route("/help-request", methods=["GET", "POST"])
def help_request():
    user = get_current_user()
    if not user:
        flash("Please login to request veterinary help.")
        return redirect(url_for("login"))

    if not INDIA_STATE_DISTRICTS:
        flash("State and district data not loaded.")
        return redirect(url_for("upload"))

    states = sorted(INDIA_STATE_DISTRICTS.keys())
    districts_by_state = INDIA_STATE_DISTRICTS

    if request.method == "POST":
        cattle_id = request.form.get("cattle_id", "").strip()
        state = request.form.get("state")
        district = request.form.get("district")

        if not cattle_id or not state or not district:
            flash("Please fill all fields.")
            return render_template(
                "help_request.html",
                user=user,
                states=states,
                districts_by_state=districts_by_state,
            )

        return redirect(url_for("nearest_vets", state=state, district=district))

    return render_template(
        "help_request.html",
        user=user,
        states=states,
        districts_by_state=districts_by_state,
    )


@app.route("/api/nearest-vets")
def api_nearest_vets():
    user = get_current_user()
    if not user or user.farm_lat is None or user.farm_lng is None:
        return jsonify({"error": "No user location set"}), 400

    centers = VetCenter.query.all()
    items = []
    for c in centers:
        d = haversine_km(user.farm_lat, user.farm_lng, c.lat, c.lng)
        items.append({
            "id": c.id,
            "name": c.name,
            "lat": c.lat,
            "lng": c.lng,
            "phone": c.phone,
            "opening_hours": c.opening_hours,
            "state": c.state,
            "district": c.district,
            "center_type": c.center_type,
            "distance_km": round(d, 1),
        })

    items.sort(key=lambda x: x["distance_km"])
    return jsonify(items[:10])


@app.route("/nearest-vets")
def nearest_vets():
    user = get_current_user()
    if not user:
        flash("Please login to view nearby veterinarians.")
        return redirect(url_for("login"))

    state = request.args.get("state")
    district = request.args.get("district")

    if not state or not district:
        flash("Please select state and district first.")
        return redirect(url_for("help_request"))

    centers = VetCenter.query.filter_by(state=state, district=district).all()

    centers_data = [
        {
            "id": c.id,
            "name": c.name,
            "phone": c.phone,
            "opening_hours": c.opening_hours,
            "state": c.state,
            "district": c.district,
            "center_type": c.center_type,
        }
        for c in centers
    ]

    return render_template(
        "nearest_vets.html",
        user=user,
        state=state,
        district=district,
        centers=centers_data,
    )


# ---------- NEW: Feedback API route ----------
@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json() or {}

    image_id = data.get("image_id")
    predicted_breed = data.get("predicted_breed")
    predicted_disease = data.get("predicted_disease")
    correct_breed = data.get("correct_breed")
    correct_disease = data.get("correct_disease")
    comment = data.get("comment")
    is_correct = data.get("is_correct")
    confidence = data.get("prediction_confidence")
    model_version = data.get("model_version")

    user_id = session.get("user_id")

    feedback = PredictionFeedback(
        image_id=image_id,
        user_id=user_id,
        predicted_breed=predicted_breed,
        predicted_disease=predicted_disease,
        correct_breed=correct_breed,
        correct_disease=correct_disease,
        is_correct=is_correct,
        comment=comment,
        prediction_confidence=confidence,
        model_version=model_version,
    )

    db.session.add(feedback)
    db.session.commit()

    return jsonify({"status": "ok"})


# ---------------------------------------------------
# Helper functions for network access
# ---------------------------------------------------
def get_local_ip():
    """Get the local IP address for network access."""
    try:
        # Connect to a remote server to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            # Fallback: get hostname IP
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "127.0.0.1"


def find_available_port(start_port=5500, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    # If no port found, use the default
    return start_port


# ---------------------------------------------------
# Main entry
# ---------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="127.0.0.1", port=5000, debug=True)
