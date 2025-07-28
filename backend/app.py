import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# flask setup
app = Flask(__name__)
CORS(app)

# Load model and feature template
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features_used.json")

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

# Load feature names
with open("features_used.json", "r") as f:
    features_used = json.load(f)


# helper function to normalize features
def normalize_col(name):
    return (
        name.strip().lower().replace("&", "and")
    ) 


# Categories
INDUSTRIES = [
    "Technology",
    "Healthcare",
    "Finance",
    "E-commerce",
    "Manufacturing",
    "Education",
    "Real Estate",
    "Retail",
    "Transportation",
    "Entertainment",
]

FUNDING_TYPES = [
    "Seed",
    "Convertible Note",
    "Series A",
    "Series B",
    "Series C",
    "Series D",
    "Series E",
    "Debt Financing",
    "Grant",
    "Product Crowdfunding",
    "Equity Crowdfunding",
    "Private Equity",
    "Post‑IPO Equity",
    "Angel",
    "Undisclosed",
    "Venture",
]

MARKET_CATS = [
    "Social Television",
    "Enterprise Search",
    "Reviews & Recommendations",
    "Biomass Power Generation",
    "Minerals",
    "Parenting",
    "Transaction Processing",
    "Hardware",
    "Auto",
    "Health Services Industry",
]


# Preprocess
def preprocess(data):
    row = pd.Series(0, index=features_used)

    # Set founded year
    row["founded_year"] = int(data.get("foundedYear", 0))

    # Set industry
    industry = normalize_col(data.get("industry", ""))
    if industry in row:
        row[industry] = 1

    # Funding types (as-is, no "funding_" prefix)
    for ft in data.get("fundingTypes", []):
        ft_norm = normalize_col(ft)
        if ft_norm in row:
            row[ft_norm] = 1

    # Market (example: "Hardware" → "market_ Hardware ")
    market = "market_ " + data.get("market", "")
    if market in row:
        row[market] = 1

    return pd.DataFrame([row])


# predict route
@app.route("/predict", methods=["POST"])
def predict():
    content = request.get_json()
    if not content:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        features = preprocess(content)
        prediction = model.predict(features)[0]
        return jsonify({"success_score": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
