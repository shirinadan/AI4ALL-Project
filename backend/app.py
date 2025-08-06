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

with bz2.BZ2File(os.path.join(BASE_DIR, "vv5_success_model_compressed.pkl"), "rb") as f:
    model = joblib.load(f)

with open(os.path.join(BASE_DIR, "vv5_features.json"), "r") as f:
    feature_names = json.load(f)


def preprocess_input(user_input_json: dict) -> pd.DataFrame:
    # Convert incoming JSON to DataFrame
    input_df = pd.DataFrame([user_input_json])

    # Handle missing categorical columns
    categorical_cols = ["market", "country_code", "state_code"]
    for col in categorical_cols:
        if col not in input_df.columns:
            input_df[col] = "Unknown"
        else:
            input_df[col] = input_df[col].fillna("Unknown")

    # One-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(pd.DataFrame(columns=categorical_cols))  # Fit structure only
    encoded_df = pd.DataFrame(
        encoder.transform(input_df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Drop original categorical columns and combine
    input_df = input_df.drop(columns=categorical_cols)
    final_input = pd.concat([input_df, encoded_df], axis=1)

    # Align columns with training features
    for col in feature_names:
        if col not in final_input.columns:
            final_input[col] = 0  # fill missing with 0

    final_input = final_input[feature_names]  # ensure correct column order
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

