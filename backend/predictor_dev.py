import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
if not API_KEY:
    print("ERROR: FOOTBALL_API_KEY not found")
    exit(1)


class LaLigaPredictor:
    def __init__(self):
        self.matches_df = None

    def fetch_data(self, seasons=["2023", "2024", "2025"]):
        headers = {"X-Auth-Token": API_KEY}
        all_matches = []

        for season in seasons:
            url = f"https://api.football-data.org/v4/competitions/2014/matches?season={season}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"Failed to fetch {season}: {response.status_code}")
                if response.status_code == 429:
                    time.sleep(5)
                continue

            for match in response.json()["matches"]:
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
                    }
                )

        self.matches_df = (
            pd.DataFrame(all_matches).sort_values("date").reset_index(drop=True)
        )
        return self

    def calculate_stats(self):
        # Rolling team stats with ELO-style rating
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
                "elo": 1500,  # Starting ELO rating
                "goals_last_6": [],  # Goals scored in last 6 matches
                "conceded_last_6": [],  # Goals conceded in last 6 matches
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
            "home_avg_goals_scored",
            "away_avg_goals_scored",
            "home_avg_goals_conceded",
            "away_avg_goals_conceded",
        ]
        for col in cols:
            self.matches_df[col] = np.nan

        for idx, row in self.matches_df.iterrows():
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]

            # Store stats before match
            h = stats[home_id]
            a = stats[away_id]

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

            # ELO ratings
            self.matches_df.at[idx, "home_elo"] = h["elo"]
            self.matches_df.at[idx, "away_elo"] = a["elo"]

            # Average goals scored/conceded in last 6 matches
            self.matches_df.at[idx, "home_avg_goals_scored"] = (
                np.mean(h["goals_last_6"]) if h["goals_last_6"] else 1.2
            )
            self.matches_df.at[idx, "away_avg_goals_scored"] = (
                np.mean(a["goals_last_6"]) if a["goals_last_6"] else 1.2
            )
            self.matches_df.at[idx, "home_avg_goals_conceded"] = (
                np.mean(h["conceded_last_6"]) if h["conceded_last_6"] else 1.2
            )
            self.matches_df.at[idx, "away_avg_goals_conceded"] = (
                np.mean(a["conceded_last_6"]) if a["conceded_last_6"] else 1.2
            )

            # Update stats after match (if completed)
            if pd.notna(row["result"]):
                h["matches"] += 1
                a["matches"] += 1
                h["home_matches"] += 1
                a["away_matches"] += 1
                h["gf"] += row["home_score"]
                h["ga"] += row["away_score"]
                a["gf"] += row["away_score"]
                a["ga"] += row["home_score"]

                # Track goals in last 6 matches
                h["goals_last_6"].append(row["home_score"])
                h["conceded_last_6"].append(row["away_score"])
                a["goals_last_6"].append(row["away_score"])
                a["conceded_last_6"].append(row["home_score"])

                if len(h["goals_last_6"]) > 6:
                    h["goals_last_6"] = h["goals_last_6"][-6:]
                if len(h["conceded_last_6"]) > 6:
                    h["conceded_last_6"] = h["conceded_last_6"][-6:]
                if len(a["goals_last_6"]) > 6:
                    a["goals_last_6"] = a["goals_last_6"][-6:]
                if len(a["conceded_last_6"]) > 6:
                    a["conceded_last_6"] = a["conceded_last_6"][-6:]

                # Update ELO ratings
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

                # Update ELO with K=32
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
        X["form_diff"] = df["home_form"].fillna(1.0) - df["away_form"].fillna(1.0)

        # ELO difference - proven most effective feature
        X["elo_diff"] = df["home_elo"].fillna(1500) - df["away_elo"].fillna(1500)

        # Attack vs Defense matchup
        X["attack_vs_defense"] = df["home_avg_goals_scored"].fillna(1.2) - df[
            "away_avg_goals_conceded"
        ].fillna(1.2)
        X["defense_vs_attack"] = df["away_avg_goals_scored"].fillna(1.2) - df[
            "home_avg_goals_conceded"
        ].fillna(1.2)

        X["strength"] = (
            X["ppg_diff"] * 0.3
            + X["home_advantage"] * 0.25
            + X["form_diff"] * 0.2
            + X["elo_diff"] * 0.01
            + X["attack_vs_defense"] * 0.125
            + X["defense_vs_attack"] * 0.125
        )
        X["matchday"] = df["matchday"]
        return X

    def test(
        self, train_seasons=["2023", "2024"], test_season="2025", test_matchdays=None
    ):
        # Split data
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
            print("No test data")
            return

        # Target: 0=away, 1=home, 2=draw
        target_map = {"HOME_TEAM": 1, "AWAY_TEAM": 0, "DRAW": 2}

        X_train = self.create_features(train_df)
        y_train = train_df["result"].map(target_map)
        X_test = self.create_features(test_df)
        y_test = test_df["result"].map(target_map)

        # Train
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train) * 100
        test_acc = model.score(X_test, y_test) * 100
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        print(f"\nTrain: {len(train_df)} matches | {train_acc:.1f}% acc")
        print(f"CV: {cv_scores.mean()*100:.1f}% (±{cv_scores.std()*100:.1f}%)")
        print(f"Test: {len(test_df)} matches | {test_acc:.1f}% acc")

        # Per-outcome breakdown
        preds = model.predict(X_test)
        for outcome, label in [(1, "Home"), (0, "Away"), (2, "Draw")]:
            mask = y_test == outcome
            if mask.sum() > 0:
                acc = (preds[mask] == outcome).sum() / mask.sum() * 100
                print(f"  {label}: {mask.sum()} matches, {acc:.1f}% acc")

        # Show sample predictions
        print(f"\nSample predictions (first 10):")
        pred_map = {0: "A", 1: "H", 2: "D"}

        for i, (idx, row) in enumerate(test_df.head(10).iterrows()):
            pred = pred_map[preds[i]]
            actual = pred_map[y_test.iloc[i]]
            check = "✓" if pred == actual else "✗"
            print(
                f"{row['home_team'][:15]:15} {int(row['home_score'])}-{int(row['away_score'])} {row['away_team'][:15]:15} | Pred:{pred} Act:{actual} {check}"
            )

        return test_acc


if __name__ == "__main__":
    predictor = LaLigaPredictor()
    predictor.fetch_data().calculate_stats()

    # Test on first 15 matchdays of 2025
    predictor.test(
        train_seasons=["2023", "2024"],
        test_season="2025",
        test_matchdays=list(range(1, 16)),
    )
