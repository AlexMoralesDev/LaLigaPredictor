import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from supabase import Client, create_client

warnings.filterwarnings("ignore")

load_dotenv()
API_KEY = os.getenv("FOOTBALL_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not API_KEY:
    print("ERROR: FOOTBALL_API_KEY not found in .env file")
    exit(1)
if not SUPABASE_URL:
    print("ERROR: SUPABASE_URL not found in .env file")
    exit(1)
if not SUPABASE_KEY:
    print("ERROR: SUPABASE_KEY not found in .env file")
    exit(1)

headers = {"X-Auth-Token": API_KEY}

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client initialized")
except Exception as e:
    print(f"ERROR: Failed to initialize Supabase client: {e}")
    exit(1)


class LaLigaPredictor:
    def __init__(self):
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

    def get_la_liga_data(self):
        print("Fetching data from API...")

        # Fetch historical data for training (2000-2024) plus current season (2025)
        seasons = ["2023", "2024", "2025"]

        urls = {
            "2023": "https://api.football-data.org/v4/competitions/2014/matches?season=2023",
            "2024": "https://api.football-data.org/v4/competitions/2014/matches?season=2024",
            "2025": "https://api.football-data.org/v4/competitions/2014/matches?season=2025",
        }

        responses = {}

        for season, url in urls.items():
            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    print(
                        f"  Fetching {season} season data (attempt {attempt + 1}/{max_retries})..."
                    )
                    response = requests.get(url, headers=headers, timeout=10)

                    if response.status_code == 200:
                        responses[season] = response.json()
                        print(f"  ✓ Successfully fetched {season} season data")
                        break
                    elif response.status_code == 429:
                        print(f"  Rate limit hit. Waiting {retry_delay * 2} seconds...")
                        time.sleep(retry_delay * 2)
                    elif response.status_code == 403:
                        print(f"  ERROR: Access forbidden. Check your API key.")
                        exit(1)
                    else:
                        print(f"  Warning: Got status code {response.status_code}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)

                except requests.exceptions.ConnectionError as e:
                    print(f"  ERROR: Connection failed - {str(e)[:100]}")
                    print(f"  This usually means:")
                    print(f"    1. No internet connection")
                    print(f"    2. DNS resolution failure")
                    print(f"    3. Firewall blocking the connection")
                    print(f"    4. VPN or proxy issues")

                    if attempt < max_retries - 1:
                        print(f"  Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"\n  Failed to connect after {max_retries} attempts.")
                        print(f"  Please check your network connection and try again.")
                        exit(1)

                except requests.exceptions.Timeout:
                    print(f"  Request timed out")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        print(f"  Failed after {max_retries} timeout attempts")
                        exit(1)

                except Exception as e:
                    print(f"  Unexpected error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise

        return responses["2023"], responses["2024"], responses["2025"]

    def create_matches_dataframe(self, data):
        matches_list = []
        for match in data["matches"]:
            ref_name = (
                match.get("referees", [{}])[0].get("name")
                if match.get("referees")
                else None
            )

            match_info = {
                "date": pd.to_datetime(match["utcDate"]),
                "matchday": match["matchday"],
                "home_team_id": match["homeTeam"]["id"],
                "away_team_id": match["awayTeam"]["id"],
                "home_team_name": match["homeTeam"]["name"],
                "away_team_name": match["awayTeam"]["name"],
                "home_score": match["score"]["fullTime"]["home"],
                "away_score": match["score"]["fullTime"]["away"],
                "result": match["score"]["winner"],
                "status": match["status"],
                "referee": ref_name,
            }
            matches_list.append(match_info)

        return pd.DataFrame(matches_list).sort_values("date")

    def calculate_stats(self, df):
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
            df[col] = np.nan

        league_avg_goals = 1.4

        for idx, row in df.iterrows():
            home_id, away_id, ref = (
                row["home_team_id"],
                row["away_team_id"],
                row["referee"],
            )
            h, a = stats[home_id], stats[away_id]

            # Store pre-match statistics
            df.at[idx, "home_ppg"] = h["points"] / max(1, h["matches"])
            df.at[idx, "away_ppg"] = a["points"] / max(1, a["matches"])
            df.at[idx, "home_gd"] = (h["gf"] - h["ga"]) / max(1, h["matches"])
            df.at[idx, "away_gd"] = (a["gf"] - a["ga"]) / max(1, a["matches"])
            df.at[idx, "home_home_ppg"] = h["home_points"] / max(1, h["home_matches"])
            df.at[idx, "away_away_ppg"] = a["away_points"] / max(1, a["away_matches"])

            df.at[idx, "home_form"] = np.mean(h["form"][-5:]) if h["form"] else 1.0
            df.at[idx, "away_form"] = np.mean(a["form"][-5:]) if a["form"] else 1.0

            df.at[idx, "home_elo"] = h["elo"]
            df.at[idx, "away_elo"] = a["elo"]

            home_attack = (h["gf"] / max(1, h["matches"])) / league_avg_goals
            away_attack = (a["gf"] / max(1, a["matches"])) / league_avg_goals
            home_defense = (h["ga"] / max(1, h["matches"])) / league_avg_goals
            away_defense = (a["ga"] / max(1, a["matches"])) / league_avg_goals

            df.at[idx, "home_attack_strength"] = (
                home_attack if h["matches"] > 0 else 1.0
            )
            df.at[idx, "away_attack_strength"] = (
                away_attack if a["matches"] > 0 else 1.0
            )
            df.at[idx, "home_defense_strength"] = (
                home_defense if h["matches"] > 0 else 1.0
            )
            df.at[idx, "away_defense_strength"] = (
                away_defense if a["matches"] > 0 else 1.0
            )

            df.at[idx, "home_form_3"] = (
                np.mean(h["form"][-3:]) if len(h["form"]) >= 3 else 1.0
            )
            df.at[idx, "away_form_3"] = (
                np.mean(a["form"][-3:]) if len(a["form"]) >= 3 else 1.0
            )
            df.at[idx, "home_form_5"] = (
                np.mean(h["form"][-5:]) if len(h["form"]) >= 5 else 1.0
            )
            df.at[idx, "away_form_5"] = (
                np.mean(a["form"][-5:]) if len(a["form"]) >= 5 else 1.0
            )

            df.at[idx, "home_goals_3"] = (
                np.mean(h["goals_last_3"]) if h["goals_last_3"] else 1.2
            )
            df.at[idx, "away_goals_3"] = (
                np.mean(a["goals_last_3"]) if a["goals_last_3"] else 1.2
            )
            df.at[idx, "home_conceded_3"] = (
                np.mean(h["conceded_last_3"]) if h["conceded_last_3"] else 1.2
            )
            df.at[idx, "away_conceded_3"] = (
                np.mean(a["conceded_last_3"]) if a["conceded_last_3"] else 1.2
            )

            df.at[idx, "home_goals_5"] = (
                np.mean(h["goals_last_5"]) if h["goals_last_5"] else 1.2
            )
            df.at[idx, "away_goals_5"] = (
                np.mean(a["goals_last_5"]) if a["goals_last_5"] else 1.2
            )
            df.at[idx, "home_conceded_5"] = (
                np.mean(h["conceded_last_5"]) if h["conceded_last_5"] else 1.2
            )
            df.at[idx, "away_conceded_5"] = (
                np.mean(a["conceded_last_5"]) if a["conceded_last_5"] else 1.2
            )

            df.at[idx, "home_goals_10"] = (
                np.mean(h["goals_last_10"]) if h["goals_last_10"] else 1.2
            )
            df.at[idx, "away_goals_10"] = (
                np.mean(a["goals_last_10"]) if a["goals_last_10"] else 1.2
            )
            df.at[idx, "home_conceded_10"] = (
                np.mean(h["conceded_last_10"]) if h["conceded_last_10"] else 1.2
            )
            df.at[idx, "away_conceded_10"] = (
                np.mean(a["conceded_last_10"]) if a["conceded_last_10"] else 1.2
            )

            # League positions
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

            df.at[idx, "home_league_position"] = position_map.get(home_id, 10)
            df.at[idx, "away_league_position"] = position_map.get(away_id, 10)

            # H2H stats
            h2h = self.h2h_stats[home_id][away_id]
            if h2h["matches"] > 0:
                total_points = h2h["home_wins"] * 3 + h2h["draws"]
                df.at[idx, "h2h_home_win_rate"] = total_points / (h2h["matches"] * 3)
                df.at[idx, "h2h_home_goals_diff"] = (
                    h2h["home_gf"] - h2h["home_ga"]
                ) / h2h["matches"]
            else:
                df.at[idx, "h2h_home_win_rate"] = 0.5
                df.at[idx, "h2h_home_goals_diff"] = 0.0

            # Referee stats
            if ref:
                ref_h, ref_a = (
                    self.ref_stats[ref][home_id],
                    self.ref_stats[ref][away_id],
                )

                if ref_h["matches"] > 0:
                    home_pts = ref_h["home_wins"] * 3 + ref_h["draws"]
                    df.at[idx, "ref_home_bias"] = home_pts / (ref_h["matches"] * 3)
                else:
                    df.at[idx, "ref_home_bias"] = 0.5

                if ref_a["matches"] > 0:
                    away_pts = ref_a["away_wins"] * 3 + ref_a["draws"]
                    df.at[idx, "ref_away_bias"] = away_pts / (ref_a["matches"] * 3)
                else:
                    df.at[idx, "ref_away_bias"] = 0.5
            else:
                df.at[idx, "ref_home_bias"] = 0.5
                df.at[idx, "ref_away_bias"] = 0.5

            # Update stats after match (only for completed matches)
            if pd.notna(row["result"]):
                h["matches"] += 1
                a["matches"] += 1
                h["home_matches"] += 1
                a["away_matches"] += 1
                h["gf"] += row["home_score"]
                h["ga"] += row["away_score"]
                a["gf"] += row["away_score"]
                a["ga"] += row["home_score"]

                # Update rolling goal stats
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

                # Update H2H
                h2h["matches"] += 1
                h2h["home_gf"] += row["home_score"]
                h2h["home_ga"] += row["away_score"]
                if row["result"] == "HOME_TEAM":
                    h2h["home_wins"] += 1
                elif row["result"] == "AWAY_TEAM":
                    h2h["away_wins"] += 1
                else:
                    h2h["draws"] += 1

                # Update referee stats
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

                # Update ELO
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

        return df

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

    def create_target(self, df):
        target_map = {"HOME_TEAM": 1, "AWAY_TEAM": 0, "DRAW": 2}
        return df["result"].map(target_map)

    def save_to_supabase(self, gameweek, predictions, training_accuracy):
        try:
            print(f"\nSaving predictions to Supabase...")

            supabase.table("predictions").delete().eq("gameweek", gameweek).execute()

            for pred in predictions:
                home_score_val = pred.get("home_score")
                away_score_val = pred.get("away_score")

                data = {
                    "gameweek": int(gameweek),
                    "home_team": pred["home_team"],
                    "away_team": pred["away_team"],
                    "predicted_result": pred["predicted_result"],
                    "home_prob": pred["home_prob"],
                    "away_prob": pred["away_prob"],
                    "draw_prob": pred["draw_prob"],
                    "match_date": pred["date"],
                    "actual_result": pred.get("actual_result"),
                    "home_score": (
                        int(home_score_val) if pd.notna(home_score_val) else None
                    ),
                    "away_score": (
                        int(away_score_val) if pd.notna(away_score_val) else None
                    ),
                    "is_correct": pred.get("correct"),
                    "predicted_at": datetime.now().isoformat(),
                }
                supabase.table("predictions").insert(data).execute()

            stats_data = {
                "training_accuracy": training_accuracy,
                "last_updated": datetime.now().isoformat(),
                "current_gameweek": int(gameweek),
            }

            existing_stats = (
                supabase.table("model_stats").select("id").limit(1).execute()
            )

            if existing_stats.data:
                supabase.table("model_stats").update(stats_data).eq(
                    "id", existing_stats.data[0]["id"]
                ).execute()
            else:
                supabase.table("model_stats").insert(stats_data).execute()

            print(f"✅ Successfully saved {len(predictions)} predictions to Supabase")

        except Exception as e:
            print(f"❌ Error saving to Supabase: {e}")
            print(f"   Full error details: {str(e)}")

    def run_prediction(self):
        """Main prediction method"""
        print("\n" + "=" * 60)
        print("LA LIGA PREDICTOR - Starting...")
        print("=" * 60 + "\n")

        training_2023_data, training_2024_data, current_2025_data = (
            self.get_la_liga_data()
        )

        print("\nCreating dataframes...")
        training_2023_df = self.create_matches_dataframe(training_2023_data)
        training_2024_df = self.create_matches_dataframe(training_2024_data)
        current_2025_df = self.create_matches_dataframe(current_2025_data)

        all_2025_matches = current_2025_df.copy()
        current_gameweek = None

        print("\nSearching for current gameweek...")
        for gw in sorted(all_2025_matches["matchday"].unique()):
            if gw < 7:
                continue
            gw_matches = all_2025_matches[all_2025_matches["matchday"] == gw]
            incomplete = len(gw_matches[gw_matches["result"].isna()])

            if incomplete > 0:
                current_gameweek = gw
                print(f"✓ Found gameweek {gw} with {incomplete} incomplete matches")
                break

        if current_gameweek is None:
            print("❌ No gameweek to predict - all matches may be completed")
            return

        gameweek_matches = all_2025_matches[
            all_2025_matches["matchday"] == current_gameweek
        ].copy()

        print("\nPreparing training data...")
        training_2023_df = training_2023_df[training_2023_df["result"].notna()].copy()
        training_2024_df = training_2024_df[training_2024_df["result"].notna()].copy()

        training_matches = (
            pd.concat(
                [
                    training_2023_df,
                    training_2024_df,
                    current_2025_df[
                        (current_2025_df["matchday"] < current_gameweek)
                        & (current_2025_df["result"].notna())
                    ],
                ]
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        print(f"✓ Training on {len(training_matches)} completed matches")

        print("\nCalculating team statistics with enhanced features...")
        all_data = (
            pd.concat([training_matches, gameweek_matches])
            .sort_values("date")
            .reset_index(drop=True)
        )
        all_data_with_stats = self.calculate_stats(all_data.copy())

        training_data = all_data_with_stats.iloc[: len(training_matches)].copy()
        prediction_data = all_data_with_stats.iloc[len(training_matches) :].copy()

        print("Creating enhanced features...")
        training_features = self.create_features(training_data)
        training_target = self.create_target(training_data)
        prediction_features = self.create_features(prediction_data)

        print("Training Random Forest model with optimized hyperparameters...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        )

        rf.fit(training_features, training_target)

        print("Making predictions...")
        predictions = rf.predict(prediction_features)
        probabilities = rf.predict_proba(prediction_features)

        training_accuracy = rf.score(training_features, training_target) * 100

        print(f"\n{'='*60}")
        print(f"GAMEWEEK {current_gameweek} PREDICTIONS")
        print(f"{'='*60}\n")

        matches_predictions = []

        for i, (idx, match) in enumerate(prediction_data.iterrows()):
            home_team = match["home_team_name"]
            away_team = match["away_team_name"]

            if predictions[i] == 0:
                predicted_result = f"{away_team} Win"
            elif predictions[i] == 1:
                predicted_result = f"{home_team} Win"
            else:
                predicted_result = "Draw"

            home_prob = probabilities[i][1] * 100
            away_prob = probabilities[i][0] * 100
            draw_prob = probabilities[i][2] * 100

            prediction_record = {
                "home_team": home_team,
                "away_team": away_team,
                "predicted_result": predicted_result,
                "home_prob": round(home_prob, 1),
                "away_prob": round(away_prob, 1),
                "draw_prob": round(draw_prob, 1),
                "date": match["date"].isoformat(),
            }

            if pd.notna(match["result"]):
                if match["result"] == "HOME_TEAM":
                    actual_result = f"{home_team} Win"
                elif match["result"] == "AWAY_TEAM":
                    actual_result = f"{away_team} Win"
                else:
                    actual_result = "Draw"

                prediction_record["actual_result"] = actual_result
                prediction_record["home_score"] = match["home_score"]
                prediction_record["away_score"] = match["away_score"]
                prediction_record["correct"] = predicted_result == actual_result

            matches_predictions.append(prediction_record)

            print(f"{home_team} vs {away_team}")
            print(f"Prediction: {predicted_result}")
            print(
                f"Confidence: {home_team} {home_prob:.1f}% | Draw {draw_prob:.1f}% | {away_team} {away_prob:.1f}%\n"
            )

        self.save_to_supabase(current_gameweek, matches_predictions, training_accuracy)

        print(f"\n{'='*60}")
        print(f"Training Accuracy: {training_accuracy:.1f}%")
        print(f"Predictions saved successfully!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        predictor = LaLigaPredictor()
        predictor.run_prediction()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
