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


# helper function to normalize features
def normalize_col(name):
    return name.lower().strip().replace(" ", "_").replace("-", "_").replace("&", "and")


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
    row = {col: 0 for col in FEATURE_COLUMNS}  # start with zeroed-out template

    # Handle numeric field
    try:
        row["founded_year"] = int(data.get("foundedYear", 0))
    except:
        row["founded_year"] = 0

    # One-hot encode: industry
    user_ind = normalize_col(data.get("industry", ""))
    row[f"industry_{user_ind}"] = 1

    # One-hot encode: funding
    for ftype in data.get("fundingTypes", []):
        norm_ftype = normalize_col(ftype)
        row[f"funding_{norm_ftype}"] = 1

    # One-hot encode: market
    market = normalize_col(data.get("market", ""))
    row[f"market_{market}"] = 1

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
