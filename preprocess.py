import pandas as pd


class DataPreprocessor:

    def __init__(self, results_path, ranking_files, elo_path):
        # Guardamos las rutas de los archivos necesarios
        self.results_path = results_path
        self.ranking_files = ranking_files
        self.elo_path = elo_path
        self.df = None

    def _clean_spaces(self, dataframe):
        # Limpiar espacios raros en blanco que puedan venir en las columnas de texto
        df_copy = dataframe.copy()
        text_cols = df_copy.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df_copy[col] = df_copy[col].astype(str).str.replace(
                r"\s+", "", regex=True
            )
        return df_copy

    def step_1_base_results(self):
        # Cargar resultados base y ordenar cronológicamente
        self.df = pd.read_csv(self.results_path, parse_dates=["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)

        # Determinar el resultado: 1 local, 0 empate, 2 visitante
        def calculate_result(row):
            if row["home_score"] > row["away_score"]:
                return 1
            elif row["home_score"] < row["away_score"]:
                return 2
            else:
                return 0

        self.df["result"] = self.df.apply(calculate_result, axis=1)
        self.df = self._clean_spaces(self.df)

        # Asignar peso o importancia según el tipo de torneo
        def tournament_weight(tournament):
            torneo = str(tournament)
            if "FIFAWorldCup" in torneo and "FIFAWorldCupqualification" not in torneo:
                return 4  # Mundial
            elif (
                "CopaAmérica" in torneo
                or "UEFAEuro" in torneo
                and "qualification" not in torneo
            ):
                return 3  # Eurocopa o Copa América
            elif (
                "FIFAWorldCupqualification" in torneo
                or "nationsleague" in torneo
                or "confederations" in torneo
            ):
                return 2  # Clasificatorios y oficiales menores
            elif "Friendly" in torneo:
                return 1  # Amistosos
            else:
                return 1.5  # Otros torneos de invitación

        self.df["tournament_weight"] = self.df["tournament"].apply(
            tournament_weight
        )
        return self

    def step_2_moving_averages(self):
        # Pasamos a formato largo (duplicando filas) para sacar medias móviles por equipo sin data leakage
        df_local = self.df[
            ["date", "home_team", "home_score", "away_score", "result"]
        ].rename(
            columns={
                "home_team": "team",
                "home_score": "goals_for",
                "away_score": "goals_against",
            }
        )
        df_local["points"] = df_local["result"].map({1: 3, 0: 1, 2: 0})

        df_away = self.df[
            ["date", "away_team", "away_score", "home_score", "result"]
        ].rename(
            columns={
                "away_team": "team",
                "away_score": "goals_for",
                "home_score": "goals_against",
            }
        )
        df_away["points"] = df_away["result"].map({1: 0, 0: 1, 2: 3})

        df_time = pd.concat([df_local, df_away]).sort_values("date")

        # Sacar medias de goles y puntos de los 3 partidos anteriores (usando shift)
        df_time["avg_goals_for"] = (
            df_time.groupby("team")["goals_for"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
        df_time["avg_goals_against"] = (
            df_time.groupby("team")["goals_against"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
        df_time["avg_points"] = (
            df_time.groupby("team")["points"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )

        df_clean = df_time[
            ["date", "team", "avg_goals_for", "avg_goals_against", "avg_points"]
        ].drop_duplicates(subset=["date", "team"])

        # Devolver las medias calculadas al formato original de partidos
        self.df = self.df.merge(
            df_clean,
            left_on=["date", "home_team"],
            right_on=["date", "team"],
            how="left",
        ).rename(
            columns={
                "avg_goals_for": "avg_goals_for_local",
                "avg_goals_against": "avg_goals_against_local",
                "avg_points": "avg_points_local",
            }
        ).drop(columns=["team"])

        self.df = self.df.merge(
            df_clean,
            left_on=["date", "away_team"],
            right_on=["date", "team"],
            how="left",
        ).rename(
            columns={
                "avg_goals_for": "avg_goals_for_away",
                "avg_goals_against": "avg_goals_against_away",
                "avg_points": "avg_points_away",
            }
        ).drop(columns=["team"])

        # Calcular las diferencias iniciales entre local y visitante
        self.df["diff_avg_goals_for"] = (
            self.df["avg_goals_for_local"] - self.df["avg_goals_for_away"]
        )
        self.df["diff_avg_goals_against"] = (
            self.df["avg_goals_against_local"] - self.df["avg_goals_against_away"]
        )
        self.df["diff_avg_points"] = (
            self.df["avg_points_local"] - self.df["avg_points_away"]
        )
        return self

    def step_3_fifa_rankings(self):
        # Filtramos para quedarnos con la época moderna desde 1993
        self.df = self.df[self.df["date"] >= "1993-01-01"].copy()
        self.df = self.df.sort_values("date")

        # Cargar y unir el histórico de rankings de la FIFA
        df_ranking = pd.concat(
            [pd.read_csv(f) for f in self.ranking_files], ignore_index=True
        )
        df_ranking["rank_date"] = pd.to_datetime(df_ranking["rank_date"])
        df_ranking = self._clean_spaces(df_ranking)
        df_ranking_clean = df_ranking[
            ["rank_date", "country_full", "total_points", "rank"]
        ].sort_values("rank_date")

        # Cambiar nombres para que cuadren con las cadenas del archivo FIFA
        alternative_names = {
            "UnitedStates": "USA",
            "SouthKorea": "KoreaRepublic",
            "Korea": "KoreaRepublic",
            "DRCongo": "Congo",
            "Iran": "IRIran",
            "IvoryCoast": "Côted'Ivoire",
        }
        self.df["home_team"] = self.df["home_team"].replace(alternative_names)
        self.df["away_team"] = self.df["away_team"].replace(alternative_names)

        # Cruce asof por fecha más cercana para local y visitante
        self.df = pd.merge_asof(
            self.df,
            df_ranking_clean,
            left_on="date",
            right_on="rank_date",
            left_by="home_team",
            right_by="country_full",
            direction="nearest",
        ).rename(
            columns={"total_points": "fifa_points_local", "rank": "ranking_local"}
        ).drop(columns=["rank_date", "country_full"], errors="ignore")

        self.df = pd.merge_asof(
            self.df,
            df_ranking_clean,
            left_on="date",
            right_on="rank_date",
            left_by="away_team",
            right_by="country_full",
            direction="nearest",
        ).rename(
            columns={"total_points": "fifa_points_away", "rank": "ranking_away"}
        ).drop(columns=["rank_date", "country_full"], errors="ignore")

        return self

    def step_4_elo_ratings(self):
        # Cargar las puntuaciones históricas de ELO
        df_elo = pd.read_csv(self.elo_path)
        df_elo["elo_date"] = pd.to_datetime(df_elo["date"], format="mixed")
        df_elo = self._clean_spaces(df_elo).sort_values("elo_date")

        # Modificar nombres de nuevo para que coincidan con el formato del ELO
        alternative_names_elo = {
            "USA": "UnitedStates",
            "KoreaRepublic": "SouthKorea",
            "Congo": "DemocraticRepublicOfCongo",
            "CzechRepublic": "Czechia",
            "IRIran": "Iran",
        }
        self.df["home_team"] = self.df["home_team"].replace(
            alternative_names_elo
        )
        self.df["away_team"] = self.df["away_team"].replace(
            alternative_names_elo
        )

        # Cruce por fecha más cercana para el rating ELO
        self.df = pd.merge_asof(
            self.df,
            df_elo[["elo_date", "team", "rating", "change"]],
            left_on="date",
            right_on="elo_date",
            left_by="home_team",
            right_by="team",
            direction="nearest",
        ).rename(columns={"rating": "elo_local", "change": "change_local"}).drop(columns=["elo_date", "team"], errors="ignore")

        self.df = pd.merge_asof(
            self.df,
            df_elo[["elo_date", "team", "rating", "change"]],
            left_on="date",
            right_on="elo_date",
            left_by="away_team",
            right_by="team",
            direction="nearest",
        ).rename(columns={"rating": "elo_away", "change": "change_away"}).drop(columns=["elo_date", "team"], errors="ignore")

        # Sacar las variables de diferencia definitivas para entrenar el modelo
        self.df["diff_elo"] = self.df["elo_local"] - self.df["elo_away"]
        self.df["diff_ranking"] = (
            self.df["ranking_local"] - self.df["ranking_away"]
        )
        self.df["diff_fifa_points"] = (
            self.df["fifa_points_local"] - self.df["fifa_points_away"]
        )

        return self

    def save_data(self, output_path):
        print(f"Guardando dataset final en: {output_path}")
        self.df.to_csv(output_path, index=False)


if __name__ == "__main__":
    RANKINGS = [
        "data/fifa_ranking-2023-07-20.csv",
        "data/fifa_ranking-2024-04-04.csv",
        "data/fifa_ranking-2024-06-20.csv",
    ]

    pipeline = DataPreprocessor(
        results_path="data/results.csv",
        ranking_files=RANKINGS,
        elo_path="data/eloratings.csv",
    )

    (
        pipeline.step_1_base_results()
        .step_2_moving_averages()
        .step_3_fifa_rankings()
        .step_4_elo_ratings()
        .save_data("data/processed_data_last_3.csv")
    )