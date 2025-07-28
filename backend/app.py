from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_model.pkl")
model = joblib.load(MODEL_PATH)

# Categories should match those used during training
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


def preprocess(data):
    """Convert incoming JSON into a one-hot encoded DataFrame."""
    row = {}
    # numerical feature
    row["founded_year"] = int(data.get("foundedYear", 0))

    # one-hot encode categorical fields
    for ind in INDUSTRIES:
        row[f"industry_{ind}"] = 1 if data.get("industry") == ind else 0

    funding = data.get("fundingTypes", [])
    for ft in FUNDING_TYPES:
        row[f"funding_{ft}"] = 1 if ft in funding else 0

    for m in MARKET_CATS:
        row[f"market_{m}"] = 1 if data.get("market") == m else 0

    return pd.DataFrame([row])


app = Flask(__name__)
CORS(app)


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
