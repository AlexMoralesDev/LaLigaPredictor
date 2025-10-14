import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
if not API_KEY:
    print("ERROR: API key not found")
    exit(1)


class LaLigaPredictor:
    def __init__(self):
        self.matches_df = None
        self.h2h_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "matches": 0,
                    "home_wins": 0,
                    "away_wins": 0,
                    "draws": 0,
                    "home_gf": 0,
                    "home_ga": 0,
                }
            )
        )
        self.ref_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "matches": 0,
                    "home_wins": 0,
                    "draws": 0,
                    "away_wins": 0,
                }
            )
        )

    def fetch_data(self, seasons=None):
        if seasons is None:
            seasons = [
                "2015",
                "2016",
                "2017",
                "2018",
                "2019",
                "2020",
                "2021",
                "2022",
                "2023",
                "2024",
                "2025",
            ]

        headers = {"X-Auth-Token": API_KEY}
        all_matches = []

        for season in seasons:
            url = f"https://api.football-data.org/v4/competitions/2014/matches?season={season}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                if response.status_code == 429:
                    time.sleep(5)
                continue

            for match in response.json()["matches"]:
                ref_name = (
                    match.get("referees", [{}])[0].get("name")
                    if match.get("referees")
                    else None
                )

                all_matches.append(
                    {
                        "season": season,
                        "date": pd.to_datetime(match["utcDate"]),
                        "matchday": match["matchday"],
                        "home_team_id": match["homeTeam"]["id"],
                        "away_team_id": match["awayTeam"]["id"],
                        "home_team": match["homeTeam"]["name"],
                        "away_team": match["awayTeam"]["name"],
                        "home_score": match["score"]["fullTime"]["home"],
                        "away_score": match["score"]["fullTime"]["away"],
                        "result": match["score"]["winner"],
                        "referee": ref_name,
                    }
                )

        self.matches_df = (
            pd.DataFrame(all_matches).sort_values("date").reset_index(drop=True)
        )
        return self

    def calculate_stats(self):
        stats = defaultdict(
            lambda: {
                "matches": 0,
                "points": 0,
                "gf": 0,
                "ga": 0,
                "home_matches": 0,
                "home_points": 0,
                "away_matches": 0,
                "away_points": 0,
                "form": [],
                "elo": 1500,
                "goals_last_3": [],
                "conceded_last_3": [],
                "goals_last_5": [],
                "conceded_last_5": [],
                "goals_last_10": [],
                "conceded_last_10": [],
            }
        )

        cols = [
            "home_ppg",
            "away_ppg",
            "home_gd",
            "away_gd",
            "home_home_ppg",
            "away_away_ppg",
            "home_form",
            "away_form",
            "home_elo",
            "away_elo",
            "home_attack_strength",
            "away_attack_strength",
            "home_defense_strength",
            "away_defense_strength",
            "home_form_3",
            "away_form_3",
            "home_form_5",
            "away_form_5",
            "home_goals_3",
            "away_goals_3",
            "home_conceded_3",
            "away_conceded_3",
            "home_goals_5",
            "away_goals_5",
            "home_conceded_5",
            "away_conceded_5",
            "home_goals_10",
            "away_goals_10",
            "home_conceded_10",
            "away_conceded_10",
            "home_league_position",
            "away_league_position",
            "h2h_home_win_rate",
            "h2h_home_goals_diff",
            "ref_home_bias",
            "ref_away_bias",
        ]
        for col in cols:
            self.matches_df[col] = np.nan

        league_avg_goals = 1.4

        for idx, row in self.matches_df.iterrows():
            home_id, away_id, ref = (
                row["home_team_id"],
                row["away_team_id"],
                row["referee"],
            )
            h, a = stats[home_id], stats[away_id]

            self.matches_df.at[idx, "home_ppg"] = h["points"] / max(1, h["matches"])
            self.matches_df.at[idx, "away_ppg"] = a["points"] / max(1, a["matches"])
            self.matches_df.at[idx, "home_gd"] = (h["gf"] - h["ga"]) / max(
                1, h["matches"]
            )
            self.matches_df.at[idx, "away_gd"] = (a["gf"] - a["ga"]) / max(
                1, a["matches"]
            )
            self.matches_df.at[idx, "home_home_ppg"] = h["home_points"] / max(
                1, h["home_matches"]
            )
            self.matches_df.at[idx, "away_away_ppg"] = a["away_points"] / max(
                1, a["away_matches"]
            )

            self.matches_df.at[idx, "home_form"] = (
                np.mean(h["form"][-5:]) if h["form"] else 1.0
            )
            self.matches_df.at[idx, "away_form"] = (
                np.mean(a["form"][-5:]) if a["form"] else 1.0
            )

            self.matches_df.at[idx, "home_elo"] = h["elo"]
            self.matches_df.at[idx, "away_elo"] = a["elo"]

            home_attack = (h["gf"] / max(1, h["matches"])) / league_avg_goals
            away_attack = (a["gf"] / max(1, a["matches"])) / league_avg_goals
            home_defense = (h["ga"] / max(1, h["matches"])) / league_avg_goals
            away_defense = (a["ga"] / max(1, a["matches"])) / league_avg_goals

            self.matches_df.at[idx, "home_attack_strength"] = (
                home_attack if h["matches"] > 0 else 1.0
            )
            self.matches_df.at[idx, "away_attack_strength"] = (
                away_attack if a["matches"] > 0 else 1.0
            )
            self.matches_df.at[idx, "home_defense_strength"] = (
                home_defense if h["matches"] > 0 else 1.0
            )
            self.matches_df.at[idx, "away_defense_strength"] = (
                away_defense if a["matches"] > 0 else 1.0
            )

            self.matches_df.at[idx, "home_form_3"] = (
                np.mean(h["form"][-3:]) if len(h["form"]) >= 3 else 1.0
            )
            self.matches_df.at[idx, "away_form_3"] = (
                np.mean(a["form"][-3:]) if len(a["form"]) >= 3 else 1.0
            )
            self.matches_df.at[idx, "home_form_5"] = (
                np.mean(h["form"][-5:]) if len(h["form"]) >= 5 else 1.0
            )
            self.matches_df.at[idx, "away_form_5"] = (
                np.mean(a["form"][-5:]) if len(a["form"]) >= 5 else 1.0
            )

            self.matches_df.at[idx, "home_goals_3"] = (
                np.mean(h["goals_last_3"]) if h["goals_last_3"] else 1.2
            )
            self.matches_df.at[idx, "away_goals_3"] = (
                np.mean(a["goals_last_3"]) if a["goals_last_3"] else 1.2
            )
            self.matches_df.at[idx, "home_conceded_3"] = (
                np.mean(h["conceded_last_3"]) if h["conceded_last_3"] else 1.2
            )
            self.matches_df.at[idx, "away_conceded_3"] = (
                np.mean(a["conceded_last_3"]) if a["conceded_last_3"] else 1.2
            )

            self.matches_df.at[idx, "home_goals_5"] = (
                np.mean(h["goals_last_5"]) if h["goals_last_5"] else 1.2
            )
            self.matches_df.at[idx, "away_goals_5"] = (
                np.mean(a["goals_last_5"]) if a["goals_last_5"] else 1.2
            )
            self.matches_df.at[idx, "home_conceded_5"] = (
                np.mean(h["conceded_last_5"]) if h["conceded_last_5"] else 1.2
            )
            self.matches_df.at[idx, "away_conceded_5"] = (
                np.mean(a["conceded_last_5"]) if a["conceded_last_5"] else 1.2
            )

            self.matches_df.at[idx, "home_goals_10"] = (
                np.mean(h["goals_last_10"]) if h["goals_last_10"] else 1.2
            )
            self.matches_df.at[idx, "away_goals_10"] = (
                np.mean(a["goals_last_10"]) if a["goals_last_10"] else 1.2
            )
            self.matches_df.at[idx, "home_conceded_10"] = (
                np.mean(h["conceded_last_10"]) if h["conceded_last_10"] else 1.2
            )
            self.matches_df.at[idx, "away_conceded_10"] = (
                np.mean(a["conceded_last_10"]) if a["conceded_last_10"] else 1.2
            )

            matchday_stats = []
            for team_id, team_stat in stats.items():
                if team_stat["matches"] > 0:
                    matchday_stats.append(
                        (
                            team_id,
                            team_stat["points"],
                            team_stat["gf"] - team_stat["ga"],
                        )
                    )

            matchday_stats.sort(key=lambda x: (-x[1], -x[2]))
            position_map = {
                team_id: idx + 1 for idx, (team_id, _, _) in enumerate(matchday_stats)
            }

            self.matches_df.at[idx, "home_league_position"] = position_map.get(
                home_id, 10
            )
            self.matches_df.at[idx, "away_league_position"] = position_map.get(
                away_id, 10
            )

            h2h = self.h2h_stats[home_id][away_id]
            if h2h["matches"] > 0:
                total_points = h2h["home_wins"] * 3 + h2h["draws"]
                self.matches_df.at[idx, "h2h_home_win_rate"] = total_points / (
                    h2h["matches"] * 3
                )
                self.matches_df.at[idx, "h2h_home_goals_diff"] = (
                    h2h["home_gf"] - h2h["home_ga"]
                ) / h2h["matches"]
            else:
                self.matches_df.at[idx, "h2h_home_win_rate"] = 0.5
                self.matches_df.at[idx, "h2h_home_goals_diff"] = 0.0

            if ref:
                ref_h, ref_a = (
                    self.ref_stats[ref][home_id],
                    self.ref_stats[ref][away_id],
                )

                if ref_h["matches"] > 0:
                    home_pts = ref_h["home_wins"] * 3 + ref_h["draws"]
                    self.matches_df.at[idx, "ref_home_bias"] = home_pts / (
                        ref_h["matches"] * 3
                    )
                else:
                    self.matches_df.at[idx, "ref_home_bias"] = 0.5

                if ref_a["matches"] > 0:
                    away_pts = ref_a["away_wins"] * 3 + ref_a["draws"]
                    self.matches_df.at[idx, "ref_away_bias"] = away_pts / (
                        ref_a["matches"] * 3
                    )
                else:
                    self.matches_df.at[idx, "ref_away_bias"] = 0.5
            else:
                self.matches_df.at[idx, "ref_home_bias"] = 0.5
                self.matches_df.at[idx, "ref_away_bias"] = 0.5

            if pd.notna(row["result"]):
                h["matches"] += 1
                a["matches"] += 1
                h["home_matches"] += 1
                a["away_matches"] += 1
                h["gf"] += row["home_score"]
                h["ga"] += row["away_score"]
                a["gf"] += row["away_score"]
                a["ga"] += row["home_score"]

                h["goals_last_3"].append(row["home_score"])
                h["conceded_last_3"].append(row["away_score"])
                a["goals_last_3"].append(row["away_score"])
                a["conceded_last_3"].append(row["home_score"])
                if len(h["goals_last_3"]) > 3:
                    h["goals_last_3"] = h["goals_last_3"][-3:]
                    h["conceded_last_3"] = h["conceded_last_3"][-3:]
                if len(a["goals_last_3"]) > 3:
                    a["goals_last_3"] = a["goals_last_3"][-3:]
                    a["conceded_last_3"] = a["conceded_last_3"][-3:]

                h["goals_last_5"].append(row["home_score"])
                h["conceded_last_5"].append(row["away_score"])
                a["goals_last_5"].append(row["away_score"])
                a["conceded_last_5"].append(row["home_score"])
                if len(h["goals_last_5"]) > 5:
                    h["goals_last_5"] = h["goals_last_5"][-5:]
                    h["conceded_last_5"] = h["conceded_last_5"][-5:]
                if len(a["goals_last_5"]) > 5:
                    a["goals_last_5"] = a["goals_last_5"][-5:]
                    a["conceded_last_5"] = a["conceded_last_5"][-5:]

                h["goals_last_10"].append(row["home_score"])
                h["conceded_last_10"].append(row["away_score"])
                a["goals_last_10"].append(row["away_score"])
                a["conceded_last_10"].append(row["home_score"])
                if len(h["goals_last_10"]) > 10:
                    h["goals_last_10"] = h["goals_last_10"][-10:]
                    h["conceded_last_10"] = h["conceded_last_10"][-10:]
                if len(a["goals_last_10"]) > 10:
                    a["goals_last_10"] = a["goals_last_10"][-10:]
                    a["conceded_last_10"] = a["conceded_last_10"][-10:]

                h2h["matches"] += 1
                h2h["home_gf"] += row["home_score"]
                h2h["home_ga"] += row["away_score"]
                if row["result"] == "HOME_TEAM":
                    h2h["home_wins"] += 1
                elif row["result"] == "AWAY_TEAM":
                    h2h["away_wins"] += 1
                else:
                    h2h["draws"] += 1

                if ref:
                    ref_h, ref_a = (
                        self.ref_stats[ref][home_id],
                        self.ref_stats[ref][away_id],
                    )
                    ref_h["matches"] += 1
                    ref_a["matches"] += 1

                    if row["result"] == "HOME_TEAM":
                        ref_h["home_wins"] += 1
                    elif row["result"] == "AWAY_TEAM":
                        ref_a["away_wins"] += 1
                    else:
                        ref_h["draws"] += 1
                        ref_a["draws"] += 1

                expected_home = 1 / (1 + 10 ** ((a["elo"] - h["elo"] - 100) / 400))

                if row["result"] == "HOME_TEAM":
                    actual_home = 1.0
                    h["points"] += 3
                    h["home_points"] += 3
                    h["form"].append(3)
                    a["form"].append(0)
                elif row["result"] == "AWAY_TEAM":
                    actual_home = 0.0
                    a["points"] += 3
                    a["away_points"] += 3
                    h["form"].append(0)
                    a["form"].append(3)
                else:
                    actual_home = 0.5
                    h["points"] += 1
                    a["points"] += 1
                    h["home_points"] += 1
                    a["away_points"] += 1
                    h["form"].append(1)
                    a["form"].append(1)

                K = 32
                h["elo"] = h["elo"] + K * (actual_home - expected_home)
                a["elo"] = a["elo"] + K * ((1 - actual_home) - (1 - expected_home))

                if len(h["form"]) > 10:
                    h["form"] = h["form"][-10:]
                if len(a["form"]) > 10:
                    a["form"] = a["form"][-10:]

        return self

    def create_features(self, df):
        X = pd.DataFrame()

        X["ppg_diff"] = df["home_ppg"].fillna(1.5) - df["away_ppg"].fillna(1.5)
        X["gd_diff"] = df["home_gd"].fillna(0) - df["away_gd"].fillna(0)
        X["home_advantage"] = df["home_home_ppg"].fillna(1.5) - df[
            "away_away_ppg"
        ].fillna(1.5)
        X["elo_diff"] = df["home_elo"].fillna(1500) - df["away_elo"].fillna(1500)

        X["xg_home"] = df["home_attack_strength"].fillna(1.0) * df[
            "away_defense_strength"
        ].fillna(1.0)
        X["xg_away"] = df["away_attack_strength"].fillna(1.0) * df[
            "home_defense_strength"
        ].fillna(1.0)
        X["xg_diff"] = X["xg_home"] - X["xg_away"]

        X["form_3_diff"] = df["home_form_3"].fillna(1.0) - df["away_form_3"].fillna(1.0)
        X["form_5_diff"] = df["home_form_5"].fillna(1.0) - df["away_form_5"].fillna(1.0)

        X["goals_3_diff"] = df["home_goals_3"].fillna(1.2) - df["away_goals_3"].fillna(
            1.2
        )
        X["goals_5_diff"] = df["home_goals_5"].fillna(1.2) - df["away_goals_5"].fillna(
            1.2
        )
        X["goals_10_diff"] = df["home_goals_10"].fillna(1.2) - df[
            "away_goals_10"
        ].fillna(1.2)

        X["conceded_3_diff"] = df["away_conceded_3"].fillna(1.2) - df[
            "home_conceded_3"
        ].fillna(1.2)
        X["conceded_5_diff"] = df["away_conceded_5"].fillna(1.2) - df[
            "home_conceded_5"
        ].fillna(1.2)
        X["conceded_10_diff"] = df["away_conceded_10"].fillna(1.2) - df[
            "home_conceded_10"
        ].fillna(1.2)

        X["position_diff"] = df["away_league_position"].fillna(10) - df[
            "home_league_position"
        ].fillna(10)

        X["home_attack_vs_away_defense"] = df["home_goals_5"].fillna(1.2) - df[
            "away_conceded_5"
        ].fillna(1.2)
        X["away_attack_vs_home_defense"] = df["away_goals_5"].fillna(1.2) - df[
            "home_conceded_5"
        ].fillna(1.2)

        X["h2h_home_win_rate"] = df["h2h_home_win_rate"].fillna(0.5)
        X["h2h_home_goals_diff"] = df["h2h_home_goals_diff"].fillna(0.0)

        X["ref_bias_diff"] = df["ref_home_bias"].fillna(0.5) - df[
            "ref_away_bias"
        ].fillna(0.5)

        X["strength"] = (
            X["ppg_diff"] * 0.20
            + X["home_advantage"] * 0.16
            + X["elo_diff"] * 0.01
            + X["xg_diff"] * 0.15
            + X["form_3_diff"] * 0.08
            + X["form_5_diff"] * 0.07
            + X["goals_5_diff"] * 0.06
            + X["conceded_5_diff"] * 0.06
            + X["position_diff"] * 0.08
            + X["home_attack_vs_away_defense"] * 0.05
            + X["away_attack_vs_home_defense"] * 0.04
            + X["h2h_home_win_rate"] * 0.03
            + X["ref_bias_diff"] * 0.01
        )

        X["matchday"] = df["matchday"]
        return X

    def test(self, train_seasons=None, test_season="2025", test_matchdays=None):
        if train_seasons is None:
            train_seasons = [
                "2015",
                "2016",
                "2017",
                "2018",
                "2019",
                "2020",
                "2021",
                "2022",
                "2023",
                "2024",
            ]

        train_df = self.matches_df[
            self.matches_df["season"].isin(train_seasons)
            & self.matches_df["result"].notna()
        ]
        test_df = self.matches_df[
            (self.matches_df["season"] == test_season)
            & self.matches_df["result"].notna()
        ]

        if test_matchdays:
            test_df = test_df[test_df["matchday"].isin(test_matchdays)]

        if len(test_df) == 0:
            print("ERROR: No test data")
            return

        target_map = {"HOME_TEAM": 1, "AWAY_TEAM": 0, "DRAW": 2}
        X_train = self.create_features(train_df)
        y_train = train_df["result"].map(target_map)
        X_test = self.create_features(test_df)
        y_test = test_df["result"].map(target_map)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        test_acc = model.score(X_test, y_test) * 100
        print(f"{test_acc:.1f}%")
        return test_acc


if __name__ == "__main__":
    predictor = LaLigaPredictor()
    predictor.fetch_data()
    predictor.calculate_stats()
    predictor.test(test_matchdays=list(range(1, 16)))
