import os
import json
import bz2
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load("vv5_success_model_compressed.pkl")


with open(os.path.join(BASE_DIR, "vv5_features.json"), "r") as f:
    feature_names = json.load(f)


def preprocess_input(user_input_json: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([user_input_json])

    # Handle missing categorical columns
    categorical_cols = ["market", "country_code", "state_code"]
    for col in categorical_cols:
        if col not in input_df.columns:
            input_df[col] = "Unknown"
        else:
            input_df[col] = input_df[col].fillna("Unknown")

    # One-hot encode manually based on vv5_features.json
    one_hot_encoded = {}
    for col in categorical_cols:
        val = input_df[col][0]
        one_hot_encoded.update({f"{col}_{val}": 1})

    # Drop original categorical columns
    input_df = input_df.drop(columns=categorical_cols)

    # Merge with one-hot values
    for key, val in one_hot_encoded.items():
        input_df[key] = val

    # Fill in any missing features from vv5_features.json
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure the order
    final_input = input_df[feature_names]
    return final_input


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.get_json()
        input_df = preprocess_input(user_input)
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": round(float(prediction), 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
