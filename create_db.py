"""
Database initialization script
Run once: python create_db.py
"""

from app import app, db
from src.models import User, Livestock, Prediction, HealthRecord
import os

def create_database():
    """Create all database tables"""
    with app.app_context():
        # Create instance folder if it doesn't exist
        os.makedirs('instance', exist_ok=True)
        
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully!")
        print("📁 Database location: instance/cattle_vision.db")
        
        # Optional: Add sample data
        add_sample_data()


def add_sample_data():
    """Add sample farmer and livestock for testing"""
    with app.app_context():
        # Check if sample data already exists
        if User.query.filter_by(email='demo@farmer.com').first():
            print("✅ Sample data already exists")
            return
        
        # Create sample farmer
        farmer = User(
            name="Demo Farmer",
            email="demo@farmer.com",
            phone="9999999999",
            role="Farmer",
            location="Sample Village",
            state="Odisha"
        )
        farmer.set_password("demo123")
        
        db.session.add(farmer)
        db.session.commit()
        
        # Create sample livestock
        livestock = Livestock(
            user_id=farmer.id,
            animal_name="Golu",
            breed="Holstein Fresian",
            animal_type="Cattle",
            unique_tag_id="TAG001",
            age_years=3.5,
            gender="Female",
            weight_kg=450,
            color="Black and White"
        )
        
        db.session.add(livestock)
        db.session.commit()
        
        print("✅ Sample data added successfully!")
        print("   Demo Farmer Email: demo@farmer.com")
        print("   Demo Password: demo123")


if __name__ == '__main__':
    create_database()
