from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    """Farmer, FLW, or Veterinarian Profile"""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(15), nullable=False)
    password_hash = db.Column(db.String(255))
    role = db.Column(db.String(50), default="Farmer")  # Farmer, FLW, Veterinarian
    location = db.Column(db.String(200))  # Village/Town
    state = db.Column(db.String(100))

    # Optional farm location (used by /api/nearest-vets)
    farm_lat = db.Column(db.Float)   # user's farm latitude
    farm_lng = db.Column(db.Float)   # user's farm longitude

    # Email verification / OTP
    is_email_verified = db.Column(db.Boolean, default=False)
    email_verified_at = db.Column(db.DateTime)
    otp_code = db.Column(db.String(6))
    otp_expires_at = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    livestock = db.relationship(
        "Livestock",
        backref="owner",
        lazy=True,
        cascade="all, delete-orphan",
    )

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.name} ({self.role})>"


class Livestock(db.Model):
    """Individual Cattle or Buffalo Record"""
    __tablename__ = "livestock"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # Basic Info
    animal_name = db.Column(db.String(100))
    breed = db.Column(db.String(100), nullable=False)
    animal_type = db.Column(db.String(50), nullable=False)  # Cattle or Buffalo
    unique_tag_id = db.Column(db.String(50), unique=True)

    # Physical Info
    age_years = db.Column(db.Float)
    gender = db.Column(db.String(20))
    weight_kg = db.Column(db.Float)
    color = db.Column(db.String(100))

    # Ownership Info
    purchase_date = db.Column(db.Date)
    purchase_price = db.Column(db.Float)

    # Status
    status = db.Column(db.String(50), default="Active")  # Active, Sold, Deceased
    last_health_check = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    predictions = db.relationship(
        "Prediction",
        backref="livestock",
        lazy=True,
        cascade="all, delete-orphan",
    )
    health_records = db.relationship(
        "HealthRecord",
        backref="livestock",
        lazy=True,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Livestock {self.animal_name or self.breed}>"


class Prediction(db.Model):
    """AI Prediction Result from Image Analysis"""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    livestock_id = db.Column(db.Integer, db.ForeignKey("livestock.id"), nullable=False, index=True)

    # Image Info
    image_filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(500))

    # Breed Prediction
    predicted_breed = db.Column(db.String(100), nullable=False)
    breed_confidence = db.Column(db.Float)  # 0‑100

    # Disease Detection
    health_status = db.Column(db.String(50))  # Healthy or Unhealthy
    disease_detected = db.Column(db.String(100))
    disease_confidence = db.Column(db.Float)  # 0‑100

    # All probabilities stored as JSON (breed_name -> prob)
    all_breed_probabilities = db.Column(db.JSON)

    # Metadata
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    health_record = db.relationship(
        "HealthRecord",
        uselist=False,
        backref="prediction",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Prediction {self.predicted_breed}>"


class HealthRecord(db.Model):
    """Disease & Treatment History"""
    __tablename__ = "health_records"

    id = db.Column(db.Integer, primary_key=True)
    livestock_id = db.Column(db.Integer, db.ForeignKey("livestock.id"), nullable=False, index=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"))

    # Disease Info
    disease_name = db.Column(db.String(150), nullable=False)
    detected_date = db.Column(db.DateTime, default=datetime.utcnow)
    severity = db.Column(db.String(20))  # Low, Medium, High
    symptoms = db.Column(db.Text)

    # Treatment Info
    treatment_given = db.Column(db.Text)
    medicine_used = db.Column(db.Text)
    veterinarian_name = db.Column(db.String(150))
    veterinarian_contact = db.Column(db.String(20))
    treatment_cost = db.Column(db.Float)

    # Notes
    veterinarian_notes = db.Column(db.Text)
    farmer_notes = db.Column(db.Text)

    # Recovery
    recovery_date = db.Column(db.Date)
    recovery_status = db.Column(db.String(50))  # Recovered, Under Treatment

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<HealthRecord {self.disease_name}>"


class VetCenter(db.Model):
    """Static vet clinic data used by nearest-vets map."""
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

    def __repr__(self):
        return f"<VetCenter {self.name} ({self.state}, {self.district})>"

