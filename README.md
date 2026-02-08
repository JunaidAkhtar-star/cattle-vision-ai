# 🐄 Cattle Vision AI

! [image alt] (https://github.com/JunaidAkhtar-star/cattle-vision-ai/blob/e5ec8c9f5b975789587751376366e483aaab2e80/Screenshot%20(262).png)

Cattle Vision AI is a Flask‑based web application that performs **cattle/buffalo breed identification** and **visual health screening** from images.  
It provides a modern UI, prediction history, Google OAuth login, and a feedback loop to improve the model over time.

---

## 📁 Project Structure

```bash
cattle_breed_image/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── Procfile                   # (Optional) For platforms like Heroku/Render
├── .gitignore                 # Git ignore rules
├── /instance                  # (Optional) Local config, DB (if used)
├── /models.py                 # SQLAlchemy models (User, Livestock, Prediction, Feedback, etc.)
├── /combined_predictor.py     # ML pipeline: breed + disease prediction
├── /disease_logic.py          # Disease risk banding, explanation, recommendation helpers
├── /auth_utils.py             # Helper: get_current_user, OTP utils, etc.
├── /config.py                 # App configuration (secret key, DB URI)
├── /static
│   ├── /uploads               # Uploaded animal images
│   ├── /css                   # Custom CSS (if any)
│   ├── /js
│   │   ├── feedback.js        # Feedback handling (correct/wrong prediction)
│   │   └── main.js            # Other front‑end JS (if any)
│   └── /img                   # Static images, logos, icons
├── /templates
│   ├── base.html              # Shared base layout
│   ├── upload.html            # Upload page (home)
│   ├── result.html            # Analysis result page (breed + disease + feedback)
│   ├── history.html           # User’s prediction history
│   ├── add_livestock.html     # Manual livestock entry
│   ├── livestock_detail.html  # Detail page per animal
│   ├── login.html             # Email/password login
│   ├── register.html          # Registration + OTP info
│   ├── verify_email.html      # Email OTP verification
│   ├── help_request.html      # Form to request nearby vets
│   └── nearest_vets.html      # List of nearest veterinary centers
├── /data
│   ├── india_states_districts.json  # State → districts mapping for forms
│   ├── vet_hosp_0.csv               # Veterinary centers data
│   └── breed_explanations.json      # (Optional) Detailed breed explanations
└── /migrations                # Alembic/Flask‑Migrate files (if used)
Note: The exact files may differ slightly; adjust the list to match your repo.

✨ Features
🔐 User authentication

Email/password registration with OTP email verification.

Google OAuth login (supports localhost, devtunnels, and production domain).

🧠 AI‑powered analysis

Cattle + buffalo breed prediction from an image.

Top‑3 breed suggestions with confidence scores.

Disease/health status assessment with risk level and recommendations.

📊 Rich UI

Tailwind‑styled result dashboard with:

Top predicted breed, explanation, species (🐄/🐃).

Pie chart of breed probabilities (Chart.js).

Detailed breed table with traits and scores.

📝 Feedback loop

Users can mark predictions as correct/incorrect.

For incorrect predictions, they can choose the correct breed and optionally specify disease and comments.

Feedback is stored via /api/feedback and associated with image_id, model version, and user (if logged in).

🐃 Livestock management

Add livestock records (ID, species, breed, age, notes).

View per‑animal prediction history.

🆘 Vet assistance

Help request form with state/district selection.

Nearby veterinary centers listing (from CSV/DB).

Optional API for nearest centers based on lat/lng.

🧪 Tech Stack
Backend: Flask, SQLAlchemy, (optional) Flask‑Migrate

Frontend: Jinja2 templates, Tailwind CSS (via CDN), Chart.js

Auth: Email + OTP, Google OAuth 2.0

Database: SQLite/PostgreSQL (depending on environment)

ML: Custom combined_predictor (breed + disease models)

⚙️ Local Development Setup
1️⃣ Clone the repo
bash
git clone https://github.com/your-username/cattle-vision-ai.git
cd cattle-vision-ai
2️⃣ Create and activate virtualenv
bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
3️⃣ Install dependencies
bash
pip install -r requirements.txt
4️⃣ Set environment variables
Create a .env file or export variables in your shell:

bash
# Example (DO NOT commit real secrets)
SECRET_KEY="your-secret-key"
DATABASE_URL="sqlite:///cattle_vision.db"  # or your Postgres URL
GOOGLE_CLIENT_ID="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"
MAIL_SERVER="smtp.example.com"
MAIL_USERNAME="your-email@example.com"
MAIL_PASSWORD="your-email-password"
Ensure app.py loads these (via os.environ or python-dotenv).

5️⃣ Initialize the database
Inside a Python shell or dedicated init script:

python
from app import app, db
with app.app_context():
    db.create_all()
Or use Flask‑Migrate if configured.

6️⃣ Run the app
bash
python app.py
# or, if using Flask CLI:
flask run
Open:

http://127.0.0.1:5000 → upload & login

http://127.0.0.1:5000/login → login page

🔑 Google OAuth Configuration
Go to Google Cloud Console → APIs & Services → Credentials.

Create an OAuth 2.0 Client ID (Web application).

Authorized redirect URIs (examples):

Local:

http://127.0.0.1:5000/login/google/callback

http://localhost:5000/login/google/callback

Devtunnels:

https://sx5b9f05-5000.inc1.devtunnels.ms/login/google/callback

Production (e.g. Render):

https://your-app.onrender.com/login/google/callback

Put GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET into environment variables.

The app’s login_with_google and google_callback routes dynamically build the correct redirect_uri based on the current host.

☁️ Deploying on Render (Example)
Push your project to GitHub.

On https://render.com:

New → Web Service → connect your repo.

Build command: pip install -r requirements.txt

Start command: gunicorn app:app

Add environment variables on Render:

SECRET_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, DB URL, etc.

After first deploy, note your public URL, e.g.:

https://cattle-vision.onrender.com

Add this redirect URI into Google Cloud:

https://cattle-vision.onrender.com/login/google/callback

Test Google login from any device.

🧬 Feedback Flow (PredictionFeedback)
When a user answers “Is this prediction correct?”:

✅ Yes: A feedback record is stored with is_correct=True.

❌ No:

User selects the correct breed from the dropdown (breed_options from breed_names in app.py).

Optionally enters correct disease and comment.

Feedback is stored via /api/feedback as a PredictionFeedback row.

This data can be used later to retrain or evaluate your models.

🚀 Roadmap Ideas
Add confidence intervals / calibration for predictions.

Admin dashboard for reviewing feedback and re‑labeling.

Multi‑image support per animal.

Support for more breeds and diseases.

📄 License
You can add your preferred license here (e.g., MIT) once you decide.

🙌 Credits

Built by Digital Dominators team as Cattle Vision AI Project to support farmers, veterinarians, and livestock managers.
