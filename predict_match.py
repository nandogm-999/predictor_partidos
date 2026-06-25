import sys
import joblib
import numpy as np
import pandas as pd

# CONFIG
DATA_PATH = "data/processed_data_last_3.csv"
MODEL_PATH = "model_GB.joblib"
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

# 1. Load the model
try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
except FileNotFoundError:
    print(
        f" Error:'{MODEL_PATH}' not found. Trun 'train.py' first."
    )
    sys.exit()


# 2. Simulation func
def simulate_match(team_A, team_B, n_steps=10000):

    row_A = df[(df["home_team"] == team_A) | (df["away_team"] == team_A)]
    row_B = df[(df["home_team"] == team_B) | (df["away_team"] == team_B)]

    if row_A.empty or row_B.empty:
        raise ValueError(
            f"Make sure the names of ({team_A} or {team_B}) are in the dataset."
        )

    row_A = row_A.iloc[-1]
    row_B = row_B.iloc[-1]

    def extract_metrics(row, team):
        if row["home_team"] == team:
            return (
                row["elo_local"],
                row["avg_goals_for_local"],
                row["avg_goals_against_local"],
                row["avg_points_local"],
                row["ranking_local"],
                row["fifa_points_local"],
                row["change_local"],
            )
        else:
            return (
                row["elo_away"],
                row["avg_goals_for_away"],
                row["avg_goals_against_away"],
                row["avg_points_away"],
                row["ranking_away"],
                row["fifa_points_away"],
                row["change_away"],
            )

    elo_A, g_f_A, g_a_A, pts_A, r_A, f_pts_A, ch_A = extract_metrics(
        row_A, team_A
    )
    elo_B, g_f_B, g_a_B, pts_B, r_B, f_pts_B, ch_B = extract_metrics(
        row_B, team_B
    )

    data_1 = {
        "diff_elo": elo_A - elo_B,
        "diff_avg_goals_for": g_f_A - g_f_B,
        "diff_avg_goals_against": g_a_A - g_a_B,
        "diff_avg_points": pts_A - pts_B,
        "diff_ranking": r_A - r_B,
        "diff_fifa_points": f_pts_A - f_pts_B,
        "ranking_local": r_A,
        "ranking_away": r_B,
        "change_local": ch_A,
        "change_away": ch_B,
    }
    data_2 = {
        "diff_elo": elo_B - elo_A,
        "diff_avg_goals_for": g_f_B - g_f_A,
        "diff_avg_goals_against": g_a_B - g_a_A,
        "diff_avg_points": pts_B - pts_A,
        "diff_ranking": r_B - r_A,
        "diff_fifa_points": f_pts_B - f_pts_A,
        "ranking_local": r_B,
        "ranking_away": r_A,
        "change_local": ch_B,
        "change_away": ch_A,
    }

    df_scenario_1 = pd.DataFrame([data_1])[FEATURES]
    df_scenario_2 = pd.DataFrame([data_2])[FEATURES]

    prob_1 = model.predict_proba(df_scenario_1)[0]
    prob_2 = model.predict_proba(df_scenario_2)[0]

    p_draw = (prob_1[0] + prob_2[0]) / 2
    p_A = (prob_1[1] + prob_2[2]) / 2
    p_B = (prob_1[2] + prob_2[1]) / 2

    total_p = p_draw + p_A + p_B
    p_draw, p_A, p_B = p_draw / total_p, p_A / total_p, p_B / total_p

    resultados_simulados = np.random.choice(
        ["Draw", f"{team_A} Wins", f"{team_B} Wins"],
        size=n_steps,
        p=[p_draw, p_A, p_B],
    )

    unique, count = np.unique(resultados_simulados, return_counts=True)
    frequencies = dict(zip(unique, count))

    print(f"\n MONTECARLO ({n_steps} iterations) | {team_A} vs {team_B}")
    for res, times in frequencies.items():
        percentage = (times / n_steps) * 100
        print(f"  {res}: {percentage:.2f}% ({times} times)")

    return frequencies



if __name__ == "__main__":
    # Match
    match = simulate_match("Spain", "France", n_steps=10000)

