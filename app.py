import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# ==========================================
# PATH CONFIG
# ==========================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Load supporting files
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
target_le = joblib.load(os.path.join(MODEL_DIR, "target_le.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
home_map = joblib.load(os.path.join(MODEL_DIR, "home_map.pkl"))
away_map = joblib.load(os.path.join(MODEL_DIR, "away_map.pkl"))

# Load classifiers
rf_clf = joblib.load(os.path.join(MODEL_DIR, "rf_clf.pkl"))
xgb_clf = joblib.load(os.path.join(MODEL_DIR, "xgb_clf.pkl"))
knn_clf = joblib.load(os.path.join(MODEL_DIR, "knn_clf.pkl"))

# Load regressors (GOALS)
rf_home = joblib.load(os.path.join(MODEL_DIR, "rf_goal_home.pkl"))
rf_away = joblib.load(os.path.join(MODEL_DIR, "rf_goal_away.pkl"))
xgb_home = joblib.load(os.path.join(MODEL_DIR, "xgb_goal_home.pkl"))
xgb_away = joblib.load(os.path.join(MODEL_DIR, "xgb_goal_away.pkl"))

# ==========================================
# FLASK APP
# ==========================================
app = Flask(__name__)

@app.route("/")
def index():
    teams = sorted(list(home_map.keys()))
    return render_template("index.html", teams=teams)

# ==========================================
# PREDICTION API
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():

    home = request.form.get("home_team")
    away = request.form.get("away_team")

    # Default match stats (bisa ditambah dari HTML nanti)
    H_shots = 10
    A_shots = 10
    H_sot = 5
    A_sot = 5
    H_fouls = 10
    A_fouls = 10
    H_corners = 3
    A_corners = 3
    H_yellow = 1
    A_yellow = 1
    H_red = 0
    A_red = 0
    season_year = 2024

    numeric = [
        H_shots, A_shots, H_sot, A_sot,
        H_fouls, A_fouls, H_corners, A_corners,
        H_yellow, A_yellow, H_red, A_red
    ]

    # ================================
    # BUILD ORDERED FEATURE VECTOR
    # ================================
    row = []
    numeric_order = [
        "H Shots","A Shots","H SOT","A SOT",
        "H Fouls","A Fouls","H Corners","A Corners",
        "H Yellow","A Yellow","H Red","A Red"
    ]

    for col in feature_cols:

        if col == "Season_year":
            row.append(season_year)

        elif col in numeric_order:
            idx = numeric_order.index(col)
            row.append(numeric[idx])

        elif col.startswith("home_prob"):
            idx = int(col.split("_")[-1])
            row.append(home_map[home][idx])

        elif col.startswith("away_prob"):
            idx = int(col.split("_")[-1])
            row.append(away_map[away][idx])

        else:
            row.append(0)

    # Convert to DataFrame (FIX WARNING)
    X = pd.DataFrame([row], columns=feature_cols)

    # =====================================
    # CLASSIFICATION
    # =====================================
    rf_pred = target_le.inverse_transform(rf_clf.predict(X))[0]
    xgb_pred = target_le.inverse_transform(xgb_clf.predict(X))[0]

    X_scaled = scaler.transform(X)
    knn_pred = target_le.inverse_transform(knn_clf.predict(X_scaled))[0]

    # =====================================
    # GOAL REGRESSION
    # =====================================
    home_rf = int(round(rf_home.predict(X)[0]))
    away_rf = int(round(rf_away.predict(X)[0]))

    home_xgb = int(round(xgb_home.predict(X)[0]))
    away_xgb = int(round(xgb_away.predict(X)[0]))

    ensemble_home = int(round((home_rf + home_xgb) / 2))
    ensemble_away = int(round((away_rf + away_xgb) / 2))

    return jsonify({
        "RandomForest": rf_pred,
        "XGBoost": xgb_pred,
        "KNN": knn_pred,
        "RF_score": f"{home_rf} - {away_rf}",
        "XGB_score": f"{home_xgb} - {away_xgb}",
        "Ensemble_score": f"{ensemble_home} - {ensemble_away}"
    })

# ==========================================
# RUN FLASK
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)
