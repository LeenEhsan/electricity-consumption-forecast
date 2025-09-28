#cd electricity_project
#python app.py
#http://127.0.0.1:5000
from flask import Flask, render_template, request, jsonify, url_for, redirect, session, flash, send_file
import os
import json
import hashlib
import requests
import pandas as pd
import numpy as np
import pickle
import joblib
import sqlite3
import io
import csv
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import pytesseract
from functools import wraps
from models import db, User, Report
from weather import get_forecast_weather_range
from fetch_weather import get_forecast_weather
from werkzeug.security import generate_password_hash, check_password_hash
import os
from utils import extract_text, classify_report_text

app = Flask(__name__)
app.secret_key = "b38b0e5e6e4417145d959465c24d0823"
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "wattcast.db")

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# Init DB
db.init_app(app)

# Load AI models safely
try:
    classifier_model = joblib.load("text_classifier.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print("Failed to load classifier/vectorizer:", e)
    classifier_model = None
    vectorizer = None

# Load main AI model
with open("power_model.pkl", "rb") as f:
    model = pickle.load(f)

# Constants
API_KEY = "b38b0e5e6e4417145d959465c24d0823"
DATA_FILE = "static/data/historical_data.csv"
SAUDI_CITIES = ["Riyadh", "Jeddah", "Macca", "Madinah", "Dammam", "Tabuk", "Abha", "Buraidah", "Hail", "Najran", "Jazan", "Al Khobar", "Al Ahsa"]
WEATHER_VIDEO_MAP = {"Clear": "clear.mp4", "Clouds": "cloudy.mp4", "Rain": "rain.mp4", "Snow": "snow.mp4"}

# DB Connection Helper
def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# Login Required Middleware
@app.before_request
def require_login():
    protected_paths = ["/dashboard_historical", "/dashboard_compare", "/dashboard_scatter", "/dashboard_future", "/dashboard_peak", "/map_forecast", "/my_reports"]
    if request.path in protected_paths and "user" not in session:
        return redirect(url_for("login"))

# Routes: Login/Register
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and user['password'] == hashed_pw:
            session['user'] = user['username']
            session['user_id'] = user['id']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        flash('Invalid credentials. Please try again.', 'danger')
        return redirect(url_for('login'))

    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    email = request.form.get('new_email')
    username = request.form.get('new_username')
    password = request.form.get('new_password')
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)", (email, username, hashed_pw))
        conn.commit()
        flash('Registration successful! You can now log in.', 'success')
    except sqlite3.IntegrityError:
        flash('Email already registered.', 'warning')
    finally:
        conn.close()

    return redirect(url_for("login"))
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not all([username, email, password]):
        return jsonify({"success": False, "error": "All fields are required."})

    existing = User.query.filter((User.username == username) | (User.email == email)).first()
    if existing:
        return jsonify({"success": False, "error": "Username or email already exists."})

    hashed_pw = generate_password_hash(password)
    new_user = User(username=username, email=email, password_hash=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"success": True})
@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"success": False, "error": "Invalid credentials."})

    session["user"] = user.username
    session["user_id"] = user.id
    return jsonify({"success": True})


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/")
def home():
    return render_template("home.html")

# Index Page
@app.route('/index')
def index():
    if 'user' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    video_file = url_for("static", filename="videos/default.mp4")
    return render_template("index.html", video_file=video_file, city=None, username=session['user'])

# Predict
@app.route('/predict', methods=["POST"])
def predict():
    city = request.form["city"]
    if city not in SAUDI_CITIES:
        return render_template("index.html", video_file=url_for('static', filename='videos/default.mp4'),
                               error="Please select a valid Saudi city.", city=None)
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()

        temp = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        wind_speed = response["wind"]["speed"]
        pressure = response["main"]["pressure"]
        clouds = response["clouds"]["all"]
        lon = response["coord"]["lon"]
        condition = response["weather"][0]["main"]

        features = [[temp, humidity, wind_speed, pressure, clouds, lon]]
        prediction = round(model.predict(features)[0], 2)

        video_filename = WEATHER_VIDEO_MAP.get(condition, "default.mp4")
        video_file = url_for("static", filename=f"videos/{video_filename}")

        return render_template("index.html",
                               city=city,
                               prediction=prediction,
                               temperature=temp,
                               humidity=humidity,
                               wind=wind_speed,
                               condition=condition,
                               video_file=video_file,
                               show_dashboard_link=True)
    except Exception:
        return render_template("index.html", video_file=url_for('static', filename='videos/default.mp4'),
                               error="Error fetching weather data.", city=None)

# Future Forecast
@app.route("/future_forecast")
def future_forecast():
    return render_template("dashboard/dashboard_future.html", cities=SAUDI_CITIES)


@app.route("/api/forecast_future", methods=["POST"])
def forecast_future():
    try:
        data = request.get_json()
        city = data.get("city")
        duration = int(data.get("duration"))

        if not city or not duration:
            return jsonify({"error": "City and duration are required."}), 400
        if duration > 16:
            return jsonify({"error": "Forecast limited to 16 days."}), 400

        start_dt = datetime.now().date()
        end_dt = start_dt + timedelta(days=duration - 1)

        forecast_data = get_forecast_weather_range(city, str(start_dt), str(end_dt))

        features = [[
            d["temp"], d["humidity"], d["wind_speed"],
            d["pressure"], d["clouds"], d["lon"]
        ] for d in forecast_data]

        predictions = model.predict(features)
        results = [{"date": d["date"], "predicted_consumption": round(predictions[i], 2)} for i, d in enumerate(forecast_data)]
        total_kwh = float(np.sum(predictions))

        return jsonify({
            "results": results,
            "metrics": {
                "cost": round(total_kwh * 0.18, 2),
                "emissions": round(total_kwh * 0.5, 2),
                "fuel": round(total_kwh * 0.05, 2)
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error occurred: {str(e)}"}), 500
    
UPLOAD_FOLDER = "static/reports"  # Ensure this folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
UPLOAD_FOLDER = "static/reports"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload_forecast_json", methods=["POST"])
def upload_forecast_json():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No selected file"}), 400

    if "user" not in session:
        return jsonify({"success": False, "message": "User not logged in"}), 401

    username = session["user"]

    # Set consistent upload path
    user_dir = os.path.join("static", "reports", username)
    os.makedirs(user_dir, exist_ok=True)

    # Save file
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit(".", 1)[-1].lower()
    file_path = os.path.join(user_dir, filename)
    file.save(file_path)

    # Extract content for classification
    try:
        content = extract_text(file_path, file_ext)

        if not content or len(content.strip()) < 10:
            predicted_category = "other"
            explanation = "Insufficient content for AI classification. Please upload a clearer file with more readable text."
        else:
            features = vectorizer.transform([content])
            predicted_category = classifier_model.predict(features)[0]
            probas = classifier_model.predict_proba(features).tolist()[0]
            explanation = ", ".join(str(round(score, 4)) for score in probas)

    except Exception as e:
        predicted_category = "other"
        explanation = str(e)

    # Ensure user exists in DB
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"success": False, "message": "User not found in database"}), 400

    # Save to database
    report = Report(
        filename=filename,
        file_type=file_ext,
        category=predicted_category,
        explanation=explanation,
        user_id=user.id
    )
    db.session.add(report)
    db.session.commit()

    # Save to metadata.json
    save_metadata(username, filename, file_type=file_ext)

    return jsonify({"success": True, "filename": filename})

@app.route("/preview_json/<filename>")
def preview_json(filename):
    user = session.get("user")
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    path = os.path.join(app.config["UPLOAD_FOLDER"], user, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return jsonify({"error": "File does not exist or is empty"}), 400

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Dashboard APIs
@app.route("/api/historical_consumption")
def historical_consumption():
    city = request.args.get("city")
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    mask = (
        (df["city"] == city)
        & (df["date"] >= from_date)
        & (df["date"] <= to_date)
    )
    filtered = df.loc[mask]

    if filtered.empty:
        return jsonify({"error": "No data found for this selection."})

    avg_consumption = float(filtered["consumption"].mean())
    latest_entry = filtered.sort_values("date", ascending=False).iloc[0]

    return jsonify({
        "labels": filtered["date"].astype(str).tolist(),
        "values": [float(v) for v in filtered["consumption"]],
        "avg_consumption": avg_consumption,
        "latest_temp": float(latest_entry["temperature"]),
        "latest_humidity": float(latest_entry["humidity"])
    })
@app.route("/generate_historical_report", methods=["POST"])
def generate_historical_report():
    city = request.form.get("city")
    from_date = request.form.get("from")
    to_date = request.form.get("to")

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    filtered = df[
        (df["city"] == city) &
        (df["date"] >= from_date) &
        (df["date"] <= to_date)
    ]

    if filtered.empty:
        flash("No data found for selected city and range.", "danger")
        return redirect(url_for("dashboard_historical"))

    avg_consumption = round(filtered["consumption"].mean(), 2)
    latest = filtered.sort_values("date", ascending=False).iloc[0]
    latest_temp = round(latest["temperature"], 2)
    latest_humidity = round(latest["humidity"], 2)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle(f"{city} – Historical Report")

    pdf.drawString(50, 750, f"Electricity Consumption Report – {city}")
    pdf.drawString(50, 720, f"From: {from_date}  To: {to_date}")
    pdf.drawString(50, 690, f"Average Consumption: {avg_consumption} kWh")
    pdf.drawString(50, 670, f"Latest Temperature: {latest_temp} °C")
    pdf.drawString(50, 650, f"Latest Humidity: {latest_humidity} %")
    pdf.drawString(50, 620, f"Total Days: {filtered.shape[0]}")

    pdf.drawString(50, 590, "Date")
    pdf.drawString(150, 590, "Consumption")
    y = 570

    for _, row in filtered.iterrows():
        if y < 100:
            pdf.showPage()
            y = 750
        pdf.drawString(50, y, row["date"].strftime("%Y-%m-%d"))
        pdf.drawString(150, y, f"{row['consumption']:.2f} kWh")
        y -= 20

    pdf.save()
    buffer.seek(0)

    # Save to disk
    username = session["user"]
    user_folder = os.path.join("static/reports", username)
    os.makedirs(user_folder, exist_ok=True)

    filename = f"{city}_historical_report.pdf"
    file_path = os.path.join(user_folder, filename)

    with open(file_path, "wb") as f:
        f.write(buffer.getbuffer())

    # === Re-extract PDF content and classify ===
    try:
        content = extract_text(file_path, "pdf")
        if not content or len(content.strip()) < 10:
            predicted_category = "other"
            explanation = "Insufficient content for AI classification."
        else:
            features = vectorizer.transform([content])
            predicted_category = classifier_model.predict(features)[0]
            probas = classifier_model.predict_proba(features).tolist()[0]
            explanation = ", ".join(str(round(score, 4)) for score in probas)
    except Exception as e:
        predicted_category = "other"
        explanation = str(e)

    # Save to database
    user = User.query.filter_by(username=username).first()
    if user:
        report = Report(
            filename=filename,
            file_type="pdf",
            category=predicted_category,
            explanation=explanation,
            user_id=user.id
        )
        db.session.add(report)
        db.session.commit()

    # Save to metadata
    save_metadata(username, filename, file_type="pdf")

    # Return PDF as response
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )

@app.route('/compare_cities')
def compare_cities():
    city1 = request.args.get("city1")
    city2 = request.args.get("city2")
    from_date = request.args.get("from")  # ✅ Missing in your code
    to_date = request.args.get("to")

    if not all([city1, city2, from_date, to_date]):
        return jsonify({"error": "Missing parameters"}), 400

    try:
        df = pd.read_csv(DATA_FILE)
        df.columns = df.columns.str.strip().str.lower()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df1 = df[(df["city"] == city1) & (df["date"] >= from_date) & (df["date"] <= to_date)].sort_values("date")
        df2 = df[(df["city"] == city2) & (df["date"] >= from_date) & (df["date"] <= to_date)].sort_values("date")

        if df1.empty or df2.empty:
            return jsonify({"error": "No data found for selected cities and range."})

        return jsonify({
            "dates": df1["date"].dt.strftime("%Y-%m-%d").tolist(),
            "city1": df1["consumption"].tolist(),
            "city2": df2["consumption"].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
#الأصلي
import hashlib

def hash_file_content(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

@app.route('/download_comparison_csv')
def download_comparison_csv():
    city1 = request.args.get("city1")
    city2 = request.args.get("city2")
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    if not all([city1, city2, from_date, to_date]):
        return "Missing required parameters.", 400

    # Generate CSV content
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df1 = df[(df["city"] == city1) & (df["date"] >= from_date) & (df["date"] <= to_date)].sort_values("date")
    df2 = df[(df["city"] == city2) & (df["date"] >= from_date) & (df["date"] <= to_date)].sort_values("date")

    if df1.empty or df2.empty:
        return "No data found for one or both cities in the given range.", 404

    merged = pd.merge(
        df1[["date", "consumption"]],
        df2[["date", "consumption"]],
        on="date",
        suffixes=(f"_{city1}", f"_{city2}")
    )

    result = pd.DataFrame({
        "Date": merged["date"].dt.strftime("%Y-%m-%d"),
        f"{city1} Consumption (kWh)": merged[f"consumption_{city1}"],
        f"{city2} Consumption (kWh)": merged[f"consumption_{city2}"]
    })

    # Convert CSV content to string
    csv_buffer = io.StringIO()
    result.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Compute a hash of the content to avoid duplicates
    content_hash = hash_file_content(csv_data)
    filename = f"{city1}_{city2}_comparison_{content_hash}.csv"

    # Save if not already stored
    username = session["user"]
    user_folder = os.path.join("static/reports", username)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, filename)

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_data)

        # Perform AI classification
        try:
            content = extract_text(file_path, "csv")
            if not content or len(content.strip()) < 10:
                predicted_category = "other"
                explanation = "Insufficient content for AI classification."
            else:
                features = vectorizer.transform([content])
                predicted_category = classifier_model.predict(features)[0]
                probas = classifier_model.predict_proba(features).tolist()[0]
                explanation = ", ".join(str(round(score, 4)) for score in probas)
        except Exception as e:
            predicted_category = "other"
            explanation = str(e)

        user = User.query.filter_by(username=username).first()
        if user and not Report.query.filter_by(user_id=user.id, filename=filename).first():
            report = Report(
                filename=filename,
                file_type="csv",
                category=predicted_category,
                explanation=explanation,
                user_id=user.id
            )
            db.session.add(report)
            db.session.commit()
            save_metadata(username, filename, file_type="csv")

    # Return file
    return send_file(file_path, mimetype="text/csv", as_attachment=True, download_name=filename)
import hashlib

def hash_file_content(content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()

@app.route("/upload_csv_report", methods=["POST"])
def upload_csv_report():
    if "user" not in session:
        return jsonify({"success": False, "message": "Not authenticated"}), 401

    username = session["user"]

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    # Read file content into string
    csv_data = file.read().decode("utf-8")
    content_hash = hash_file_content(csv_data)
    original_name = os.path.splitext(file.filename)[0]
    filename = f"{original_name}_{content_hash}.csv"

    user_folder = os.path.join(UPLOAD_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, filename)

    # Don't overwrite if same content exists
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_data)

        # === AI classification ===
        try:
            content = extract_text(file_path, "csv")
            if not content or len(content.strip()) < 10:
                predicted_category = "other"
                explanation = "Insufficient content for AI classification. File too short or unreadable."
            else:
                features = vectorizer.transform([content])
                predicted_category = classifier_model.predict(features)[0]
                probas = classifier_model.predict_proba(features).tolist()[0]
                explanation = ", ".join(str(round(score, 4)) for score in probas)
        except Exception as e:
            predicted_category = "other"
            explanation = str(e)

        # === Save report to DB if new ===
        user = User.query.filter_by(username=username).first()
        if user and not Report.query.filter_by(user_id=user.id, filename=filename).first():
            report = Report(
                filename=filename,
                file_type="csv",
                category=predicted_category,
                explanation=explanation,
                user_id=user.id
            )
            db.session.add(report)
            db.session.commit()

            save_metadata(username, filename, file_type="csv")

    return jsonify({"success": True, "filename": filename})


@app.route("/api/scatter_data")
def scatter_data():
    city = request.args.get("city")

    df = pd.read_csv("static/data/historical_data.csv")

    # Filter by city
    city_df = df[df["City"] == city]

    if city_df.empty:
        print(f"No data found for {city}")
        return jsonify({"error": "No data for selected city"}), 404

    # Convert to numeric and clean
    city_df["Temperature"] = pd.to_numeric(city_df["Temperature"], errors="coerce")
    city_df["Consumption"] = pd.to_numeric(city_df["Consumption"], errors="coerce")
    city_df = city_df.dropna(subset=["Temperature", "Consumption"])

    # Group by temperature (rounded to 1 decimal) and average the consumption
    city_df["Temperature"] = city_df["Temperature"].round(1)
    grouped = city_df.groupby("Temperature")["Consumption"].mean().reset_index()
    grouped = grouped.round({"Temperature": 1, "Consumption": 1})

    print(grouped.head())  # Debugging line

    data = grouped.to_dict(orient="records")
    return jsonify({"data": data})


@app.route("/api/peak_days")
def api_peak_days():
    try:
        city = request.args.get("city")
        if not city:
            return jsonify({"error": "City is required"}), 400

        df = pd.read_csv(DATA_FILE)
        df.columns = df.columns.str.strip()

        consumption_col = next((col for col in df.columns if "consumption" in col.lower()), None)
        if not consumption_col:
            return jsonify({"error": "Consumption column not found"}), 500

        df = df[df["City"] == city]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        top_days = df.sort_values(by=consumption_col, ascending=False).head(5)

        return jsonify({
            "dates": top_days["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": top_days[consumption_col].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast_map_data')
def forecast_map_data():
    date = request.args.get('date')
    results = []
    try:
        for city in SAUDI_CITIES:
            weather = get_forecast_weather(city, date)
            if weather:
                features = [
                    weather["temp"],
                    weather["humidity"],
                    weather["wind_speed"],
                    weather["pressure"],
                    weather["clouds"],
                    weather["lon"]
                ]
                consumption = float(model.predict([features])[0])
                results.append({
                    "name": city,
                    "lat": weather["lat"],
                    "lon": weather["lon"],
                    "consumption": round(consumption, 2)
                })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
MAP_UPLOAD_FOLDER = "static/reports"  # Folder to store images
os.makedirs(MAP_UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload_map_image", methods=["POST"])
def upload_map_image():
    if "user" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    username = session["user"]

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"forecast_map_{timestamp}.png"

    # Create the user's report folder if it doesn't exist
    user_folder = os.path.join("static", "reports", username)
    os.makedirs(user_folder, exist_ok=True)

    # Save the file
    file_path = os.path.join(user_folder, filename)
    file.save(file_path)

    # Register the file in metadata
    save_metadata(username, filename, file_type="png")

    return jsonify({"status": "success", "filename": filename})

from flask import send_from_directory

# === Helper Functions ===

def get_user_folder():
    username = session.get("user")
    folder = os.path.join("static", "reports", username)
    os.makedirs(folder, exist_ok=True)
    return folder, username

def classify_report(text):
    if classifier_model is None or vectorizer is None:
        return "unknown"
    X = vectorizer.transform([text])
    return classifier_model.predict(X)[0]

def upload_file_to_storage(file_buffer, filename, user):
    try:
        response = requests.post(
            "http://localhost:8000/api/upload",
            files={"file": (filename, file_buffer)},
            data={"user": user}
        )
        print(response.status_code, response.text)
        return response.status_code == 200
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

# === Save metadata for uploaded file ===
def save_metadata(username, filename, file_type):
    metadata_path = os.path.join("static", "reports", username, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Load existing metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"files": []}

    # === Extract file content ===
    try:
        path = os.path.join("static", "reports", username, filename)
        ext = filename.rsplit(".", 1)[-1].lower()
        content = extract_text(path, ext)
    except Exception as e:
        content = ""
        print(f"[ERROR] Failed to extract content from {filename}: {e}")

    # === AI Classification based on text ===
    try:
        category, explanation_list = classify_report_text(content, filename=filename)
    except Exception as e:
        category = "other"
        explanation_list = [f"Classification failed: {str(e)}"]

    # === Save to metadata.json ===
    new_entry = {
        "filename": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_type": file_type,
        "category": category,
        "explanation": explanation_list
    }
    metadata["files"].append(new_entry)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # === Save to database
    user = User.query.filter_by(username=username).first()
    if user:
        existing = Report.query.filter_by(user_id=user.id, filename=filename).first()
        if not existing:
            explanation_str = ", ".join(explanation_list)
            report = Report(
                filename=filename,
                file_type=file_type,
                category=category,
                explanation=explanation_str,
                user_id=user.id
            )
            db.session.add(report)
            db.session.commit()
def remove_from_metadata(username, filename):
    metadata_path = os.path.join("static", "reports", username, "metadata.json")
    if not os.path.exists(metadata_path):
        return

    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return

    data["files"] = [entry for entry in data.get("files", []) if entry.get("filename") != filename]

    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=2)

# === Flask Routes ===

@app.route("/delete_report", methods=["POST"])
def delete_report():
    if "user" not in session:
        return jsonify({"success": False, "message": "Not authenticated"}), 401

    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "message": "No filename provided"}), 400

    username = session["user"]
    file_path = os.path.join("static", "reports", username, filename)
    metadata_path = os.path.join("static", "reports", username, "metadata.json")

    try:
        # Attempt to delete the file (optional if it doesn't exist)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from metadata.json
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            metadata["files"] = [f for f in metadata["files"] if f["filename"] != filename]

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Remove from database
        user = User.query.filter_by(username=username).first()
        if user:
            report = Report.query.filter_by(user_id=user.id, filename=filename).first()
            if report:
                db.session.delete(report)
                db.session.commit()

        return jsonify({"success": True})
    except Exception as e:
        print(f"[ERROR] Failed to delete {filename}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

from flask import send_file, session, abort

@app.route("/serve_report/<filename>")
def serve_report(filename):
    if "user" not in session:
        return abort(401)

    username = session["user"]
    base_folders = ["static/reports", "uploads"]

    for folder in base_folders:
        path = os.path.join(folder, username, filename)
        if os.path.exists(path):
            ext = filename.rsplit(".", 1)[-1].lower()
            if ext in ["json", "csv", "txt"]:
                return send_file(path, mimetype="text/plain", as_attachment=False)
            elif ext in ["png", "jpg", "jpeg"]:
                return send_file(path, mimetype=f"image/{ext}", as_attachment=False)
            elif ext == "pdf":
                return send_file(path, mimetype="application/pdf", as_attachment=False)
            else:
                return send_file(path, as_attachment=True)

    return "File not found", 404

@app.route("/reclassify_reports")
def reclassify_reports():
    if "user" not in session:
        flash("Please log in to reclassify reports.", "warning")
        return redirect(url_for("login"))

    username = session["user"]
    metadata_path = os.path.join("static", "reports", username, "metadata.json")

    if not os.path.exists(metadata_path):
        return "⚠️ metadata.json not found", 404

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        return "⚠️ Invalid metadata format", 500

    for file in metadata.get("files", []):
        filename = file.get("filename", "")
        path = os.path.join("static", "reports", username, filename)
        ext = filename.rsplit(".", 1)[-1].lower()

        try:
            content = extract_text(path, ext)
            category, explanation = classify_report_text(content, filename=filename)
        except Exception as e:
            category = "other"
            explanation = [f"Reclassification failed: {str(e)}"]

        file["category"] = category
        file["explanation"] = explanation

        # Update the database too
        user = User.query.filter_by(username=username).first()
        if user:
            report = Report.query.filter_by(user_id=user.id, filename=filename).first()
            if report:
                report.category = category
                report.explanation = ", ".join(explanation)
                db.session.commit()

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    flash("✅ Reclassification completed successfully.", "success")
    return redirect(url_for("my_reports"))

# === Classification function with explanation + fallback ===

@app.route('/my_reports')
def my_reports():
    username = session.get("user")
    if not username:
        return redirect(url_for("login"))

    user = User.query.filter_by(username=username).first()
    if not user:
        return redirect(url_for("login"))

    filter_category = request.args.get("filter", "all")

    if filter_category == "all":
        files = Report.query.filter_by(user_id=user.id).order_by(Report.timestamp.desc()).all()
    else:
        files = Report.query.filter_by(user_id=user.id, category=filter_category).order_by(Report.timestamp.desc()).all()

    return render_template("my_reports.html", files=files, selected_category=filter_category)


@app.route("/download_report/<filename>")
def download_report(filename):
    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]
    user_folder = os.path.join("static", "reports", username)
    file_path = os.path.join(user_folder, filename)

    if not os.path.exists(file_path):
        flash("File not found.", "error")
        return redirect(url_for("my_reports"))

    # Determine the correct mimetype for downloading
    ext = filename.rsplit(".", 1)[-1].lower()
    mimetype = {
        "csv": "text/csv",
        "json": "application/json",
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg"
    }.get(ext, "application/octet-stream")  # fallback if unknown

    return send_from_directory(
        user_folder,
        filename,
        as_attachment=True,
        mimetype=mimetype
    )

# Dashboard Views

@app.route('/dashboard_historical')
def dashboard_historical():
    username = session.get("user")
    user = User.query.filter_by(username=username).first() if username else None
    return render_template("dashboard/dashboard_historical.html", user=user)

@app.route('/dashboard_compare')
def dashboard_compare():
    username = session.get("user")
    user = User.query.filter_by(username=username).first() if username else None
    return render_template("dashboard/dashboard_compare.html", user=user)

@app.route('/dashboard_scatter')
def dashboard_scatter():
    username = session.get("user")
    user = User.query.filter_by(username=username).first() if username else None
    return render_template("dashboard/dashboard_scatter.html", user=user)

@app.route('/dashboard_future')
def dashboard_future():
    username = session.get("user")
    user = User.query.filter_by(username=username).first() if username else None
    return render_template("dashboard/dashboard_future.html", user=user, cities=SAUDI_CITIES)

@app.route('/dashboard_peak')
def dashboard_peak():
    username = session.get("user")
    user = User.query.filter_by(username=username).first() if username else None
    return render_template("dashboard/dashboard_peak.html", user=user)

@app.route('/map_forecast')
def map_forecast():
    username = session.get("user")
    user = User.query.filter_by(username=username).first() if username else None
    return render_template("dashboard/map_forecast.html", user=user)

# Run app
if __name__ == "__main__":
    with app.app_context():
        # Optional: create tables at start
        db.create_all()
        print("Tables created in app.py")

    # Run the Flask development server and keep it running
    app.run(host='127.0.0.1', port=5000, debug=True)

