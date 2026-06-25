import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Predictors used for training the model
DATA_PATH = "data/processed_data_last_3.csv"
FEATURES = [
    "diff_elo",
    "diff_avg_goals_for",
    "diff_avg_goals_against",
    "diff_avg_points",
    "diff_ranking",
    "diff_fifa_points",
    "ranking_local",
    "ranking_away",
    "change_local",
    "change_away",
]

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).copy()

    df_train_final = df[df["date"] < "2026-06-01"].dropna(subset=FEATURES)

    X_train_final = df_train_final[FEATURES]
    y_train_final = df_train_final["result"]

    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=70,
        random_state=45,
        class_weight='balanced'
    )
    model.fit(X_train_final, y_train_final)

    joblib.dump(model, "model_GB.joblib")