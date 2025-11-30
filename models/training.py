import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ============================
# PATH
# ============================
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "/root/college/AI/footballpred/football-predictor-ai/data/EnglandCSVcleanded.csv")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Raw shape:", df.shape)

# ============================
# CLEAN SEASON
# ============================
def season_to_year(season_str):
    a, b = str(season_str).split("/")
    return 2000 + int(b)

df["Season_year"] = df["Season"].apply(season_to_year)
df = df[df["Season_year"].between(2020, 2025)]
print("Filtered rows:", df.shape)

# ============================
# TARGET LABEL (H/D/A)
# ============================
le = LabelEncoder()
df["target"] = le.fit_transform(df["FT Result"])
print("Classes:", le.classes_)

# ============================
# HOME/AWAY PROB FEATURES
# ============================
teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))

home_map = {}
away_map = {}

for t in teams:
    # home perf
    subset_h = df[df["HomeTeam"] == t]["FT Result"].value_counts(normalize=True)
    home_map[t] = np.array([subset_h.get(c, 0.0) for c in le.classes_])

    # away perf
    subset_a = df[df["AwayTeam"] == t]["FT Result"].value_counts(normalize=True)
    away_map[t] = np.array([subset_a.get(c, 0.0) for c in le.classes_])

home_cols = [f"home_prob_{i}" for i in range(len(le.classes_))]
away_cols = [f"away_prob_{i}" for i in range(len(le.classes_))]

home_prob_arr = []
away_prob_arr = []

for _, row in df.iterrows():
    home_prob_arr.append(home_map[row["HomeTeam"]])
    away_prob_arr.append(away_map[row["AwayTeam"]])

df[home_cols] = np.array(home_prob_arr)
df[away_cols] = np.array(away_prob_arr)

# ============================
# NUMERIC FEATURES
# ============================
numeric_cols = [
    'H Shots','A Shots','H SOT','A SOT',
    'H Fouls','A Fouls','H Corners','A Corners',
    'H Yellow','A Yellow','H Red','A Red'
]

numeric_cols = [c for c in numeric_cols if c in df.columns]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

# ============================
# FINAL FEATURES
# ============================
feature_cols = ["Season_year"] + numeric_cols + home_cols + away_cols

X = df[feature_cols]
y = df["target"]

print("X shape:", X.shape)

# ============================
# SPLIT + SCALE
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# CLASSIFICATION MODELS (H/D/A)
# ============================
print("\nTraining classifiers...")

rf_clf = RandomForestClassifier(n_estimators=400, random_state=42)
rf_clf.fit(X_train, y_train)

xgb_clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)
xgb_clf.fit(X_train, y_train)

knn_clf = KNeighborsClassifier(n_neighbors=7)
knn_clf.fit(X_train_scaled, y_train)

print("RF acc:", accuracy_score(y_test, rf_clf.predict(X_test)))
print("XGB acc:", accuracy_score(y_test, xgb_clf.predict(X_test)))
print("KNN acc:", accuracy_score(y_test, knn_clf.predict(X_test_scaled)))

# ============================
# REGRESSION MODELS (PREDICT GOALS)
# ============================
print("\nTraining Goal Regressors...")

y_home_goal = df["FTH Goals"]
y_away_goal = df["FTA Goals"]

X_train_r, X_test_r, yh_train, yh_test = train_test_split(
    X, y_home_goal, test_size=0.2, random_state=42
)
_, _, ya_train, ya_test = train_test_split(
    X, y_away_goal, test_size=0.2, random_state=42
)

rf_reg_home = RandomForestRegressor(n_estimators=400, random_state=42)
rf_reg_away = RandomForestRegressor(n_estimators=400, random_state=42)

rf_reg_home.fit(X_train_r, yh_train)
rf_reg_away.fit(X_train_r, ya_train)

xgb_reg_home = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6)
xgb_reg_away = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6)

xgb_reg_home.fit(X_train_r, yh_train)
xgb_reg_away.fit(X_train_r, ya_train)

print("RF Home MAE:", mean_absolute_error(yh_test, rf_reg_home.predict(X_test_r)))
print("RF Away MAE:", mean_absolute_error(ya_test, rf_reg_away.predict(X_test_r)))
print("XGB Home MAE:", mean_absolute_error(yh_test, xgb_reg_home.predict(X_test_r)))
print("XGB Away MAE:", mean_absolute_error(ya_test, xgb_reg_away.predict(X_test_r)))

# ============================
# SAVE MODELS
# ============================
print("\nSaving models...")

joblib.dump(rf_clf, os.path.join(SAVE_DIR, "rf_clf.pkl"))
joblib.dump(xgb_clf, os.path.join(SAVE_DIR, "xgb_clf.pkl"))
joblib.dump(knn_clf, os.path.join(SAVE_DIR, "knn_clf.pkl"))

joblib.dump(rf_reg_home, os.path.join(SAVE_DIR, "rf_goal_home.pkl"))
joblib.dump(rf_reg_away, os.path.join(SAVE_DIR, "rf_goal_away.pkl"))
joblib.dump(xgb_reg_home, os.path.join(SAVE_DIR, "xgb_goal_home.pkl"))
joblib.dump(xgb_reg_away, os.path.join(SAVE_DIR, "xgb_goal_away.pkl"))

joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(SAVE_DIR, "target_le.pkl"))
joblib.dump(feature_cols, os.path.join(SAVE_DIR, "feature_cols.pkl"))
joblib.dump(home_map, os.path.join(SAVE_DIR, "home_map.pkl"))
joblib.dump(away_map, os.path.join(SAVE_DIR, "away_map.pkl"))

print("Training completed successfully!")