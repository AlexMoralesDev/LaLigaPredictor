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
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

load_dotenv()
API_KEY = os.getenv("FOOTBALL_API_KEY")

if not API_KEY:
    print("ERROR: FOOTBALL_API_KEY not found in .env file")
    exit(1)

headers = {"X-Auth-Token": API_KEY}


class SimplePredictor:
    """Simplified La Liga predictor for testing features"""

    def __init__(self, seasons=["2023", "2024", "2025"]):
        self.seasons = seasons
        self.raw_data = {}
        self.matches_df = None
        self.team_stats = None

    def fetch_data(self):
        """Fetch data from API"""
        print("Fetching data from API...")

        for season in self.seasons:
            url = f"https://api.football-data.org/v4/competitions/2014/matches?season={season}"

            try:
                print(f"  Fetching {season} season...")
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    self.raw_data[season] = response.json()
                    print(f"  ✓ {season} season fetched")
                elif response.status_code == 429:
                    print(f"  Rate limit hit. Waiting...")
                    time.sleep(5)
                else:
                    print(f"  Warning: Status code {response.status_code}")

            except Exception as e:
                print(f"  Error: {e}")
                raise

        print("✓ All data fetched\n")
        return self

    def prepare_matches(self):
        """Convert raw data to DataFrame"""
        print("Preparing matches DataFrame...")

        matches_list = []
        for season, data in self.raw_data.items():
            for match in data["matches"]:
                matches_list.append(
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
                        "status": match["status"],
                    }
                )

        self.matches_df = (
            pd.DataFrame(matches_list).sort_values("date").reset_index(drop=True)
        )

        completed = len(self.matches_df[self.matches_df["result"].notna()])
        upcoming = len(self.matches_df[self.matches_df["result"].isna()])

        print(
            f"✓ Prepared {len(self.matches_df)} matches ({completed} completed, {upcoming} upcoming)\n"
        )
        return self

    def calculate_team_stats(self):
        """Calculate rolling team statistics"""
        print("Calculating team statistics...")

        stats = defaultdict(
            lambda: {
                "matches": 0,
                "points": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "home_points": 0,
                "away_points": 0,
                "home_matches": 0,
                "away_matches": 0,
                "recent_form": [],  # Last 5 results (3=win, 1=draw, 0=loss)
            }
        )

        # Add columns for stats at time of match
        stat_columns = [
            "home_ppg",
            "away_ppg",
            "home_gd",
            "away_gd",
            "home_home_ppg",
            "away_away_ppg",
            "home_form",
            "away_form",
        ]

        for col in stat_columns:
            self.matches_df[col] = np.nan

        # Calculate stats chronologically
        for idx, match in self.matches_df.iterrows():
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]

            # Store stats BEFORE this match
            home_stats = stats[home_id]
            away_stats = stats[away_id]

            self.matches_df.at[idx, "home_ppg"] = home_stats["points"] / max(
                1, home_stats["matches"]
            )
            self.matches_df.at[idx, "away_ppg"] = away_stats["points"] / max(
                1, away_stats["matches"]
            )

            self.matches_df.at[idx, "home_gd"] = (
                home_stats["goals_for"] - home_stats["goals_against"]
            ) / max(1, home_stats["matches"])
            self.matches_df.at[idx, "away_gd"] = (
                away_stats["goals_for"] - away_stats["goals_against"]
            ) / max(1, away_stats["matches"])

            self.matches_df.at[idx, "home_home_ppg"] = home_stats["home_points"] / max(
                1, home_stats["home_matches"]
            )
            self.matches_df.at[idx, "away_away_ppg"] = away_stats["away_points"] / max(
                1, away_stats["away_matches"]
            )

            recent_home = (
                home_stats["recent_form"][-5:] if home_stats["recent_form"] else [1.0]
            )
            recent_away = (
                away_stats["recent_form"][-5:] if away_stats["recent_form"] else [1.0]
            )
            self.matches_df.at[idx, "home_form"] = np.mean(recent_home)
            self.matches_df.at[idx, "away_form"] = np.mean(recent_away)

            # Update stats AFTER this match (if completed)
            if pd.notna(match["result"]):
                home_score = match["home_score"]
                away_score = match["away_score"]

                # Update match counts
                stats[home_id]["matches"] += 1
                stats[away_id]["matches"] += 1
                stats[home_id]["home_matches"] += 1
                stats[away_id]["away_matches"] += 1

                # Update goals
                stats[home_id]["goals_for"] += home_score
                stats[home_id]["goals_against"] += away_score
                stats[away_id]["goals_for"] += away_score
                stats[away_id]["goals_against"] += home_score

                # Update points and form
                if match["result"] == "HOME_TEAM":
                    stats[home_id]["wins"] += 1
                    stats[home_id]["points"] += 3
                    stats[home_id]["home_points"] += 3
                    stats[away_id]["losses"] += 1
                    stats[home_id]["recent_form"].append(3)
                    stats[away_id]["recent_form"].append(0)
                elif match["result"] == "AWAY_TEAM":
                    stats[away_id]["wins"] += 1
                    stats[away_id]["points"] += 3
                    stats[away_id]["away_points"] += 3
                    stats[home_id]["losses"] += 1
                    stats[home_id]["recent_form"].append(0)
                    stats[away_id]["recent_form"].append(3)
                else:  # DRAW
                    stats[home_id]["draws"] += 1
                    stats[away_id]["draws"] += 1
                    stats[home_id]["points"] += 1
                    stats[away_id]["points"] += 1
                    stats[home_id]["home_points"] += 1
                    stats[away_id]["away_points"] += 1
                    stats[home_id]["recent_form"].append(1)
                    stats[away_id]["recent_form"].append(1)

                # Keep only last 10 results
                if len(stats[home_id]["recent_form"]) > 10:
                    stats[home_id]["recent_form"] = stats[home_id]["recent_form"][-10:]
                if len(stats[away_id]["recent_form"]) > 10:
                    stats[away_id]["recent_form"] = stats[away_id]["recent_form"][-10:]

        self.team_stats = stats
        print(f"✓ Statistics calculated for {len(stats)} teams\n")
        return self

    def create_features(self, df):
        """Create feature set for modeling"""
        features = pd.DataFrame(index=df.index)

        # Core strength features
        features["ppg_diff"] = df["home_ppg"].fillna(1.5) - df["away_ppg"].fillna(1.5)
        features["gd_diff"] = df["home_gd"].fillna(0) - df["away_gd"].fillna(0)
        features["home_advantage"] = df["home_home_ppg"].fillna(1.5) - df[
            "away_away_ppg"
        ].fillna(1.5)

        # Form features
        features["form_diff"] = df["home_form"].fillna(1.0) - df["away_form"].fillna(
            1.0
        )

        # Combined features
        features["overall_strength"] = (
            features["ppg_diff"] * 0.4
            + features["home_advantage"] * 0.3
            + features["form_diff"] * 0.3
        )

        # Match context
        features["matchday"] = df["matchday"]
        features["is_early_season"] = (df["matchday"] <= 8).astype(int)

        return features

    def create_target(self, df):
        """Create target variable (0=away win, 1=home win, 2=draw)"""
        target_map = {"HOME_TEAM": 1, "AWAY_TEAM": 0, "DRAW": 2}
        return df["result"].map(target_map)

    def train_model(self, train_df):
        """Train model on training data"""
        X_train = self.create_features(train_df)
        y_train = self.create_target(train_df)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, train_df, test_df):
        """Evaluate model on both training and test sets"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60 + "\n")

        # Training performance
        X_train = self.create_features(train_df)
        y_train = self.create_target(train_df)
        train_acc = model.score(X_train, y_train) * 100
        print(f"Training Set:")
        print(f"  Matches: {len(train_df)}")
        print(f"  Accuracy: {train_acc:.1f}%\n")

        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-Validation (5-fold):")
        print(f"  Mean: {cv_scores.mean()*100:.1f}%")
        print(f"  Std: ±{cv_scores.std()*100:.1f}%\n")

        # Test set performance
        X_test = self.create_features(test_df)
        y_test = self.create_target(test_df)

        predictions = model.predict(X_test)
        test_acc = model.score(X_test, y_test) * 100

        # Calculate per-outcome accuracy
        home_wins = y_test == 1
        away_wins = y_test == 0
        draws = y_test == 2

        home_acc = (
            (predictions[home_wins] == 1).sum() / max(1, home_wins.sum()) * 100
            if home_wins.sum() > 0
            else 0
        )
        away_acc = (
            (predictions[away_wins] == 0).sum() / max(1, away_wins.sum()) * 100
            if away_wins.sum() > 0
            else 0
        )
        draw_acc = (
            (predictions[draws] == 2).sum() / max(1, draws.sum()) * 100
            if draws.sum() > 0
            else 0
        )

        print(f"Test Set:")
        print(f"  Matches: {len(test_df)}")
        print(f"  Overall Accuracy: {test_acc:.1f}%")
        print(f"\n  Breakdown:")
        print(f"    Home Wins: {home_wins.sum()} matches - {home_acc:.1f}% accuracy")
        print(f"    Away Wins: {away_wins.sum()} matches - {away_acc:.1f}% accuracy")
        print(f"    Draws: {draws.sum()} matches - {draw_acc:.1f}% accuracy")

        print(f"\n{'='*60}\n")

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "cv_mean": cv_scores.mean() * 100,
            "cv_std": cv_scores.std() * 100,
            "home_accuracy": home_acc,
            "away_accuracy": away_acc,
            "draw_accuracy": draw_acc,
        }

    def test_on_completed_matches(
        self, train_seasons=["2023", "2024"], test_season="2025", test_matchdays=None
    ):
        """
        Test the model on already completed matches

        Args:
            train_seasons: List of seasons to train on
            test_season: Season to test on
            test_matchdays: Specific matchdays to test (None = all completed)
        """
        print("\n" + "=" * 60)
        print("TESTING ON COMPLETED MATCHES")
        print("=" * 60 + "\n")

        # Get training data
        train_matches = self.matches_df[
            self.matches_df["season"].isin(train_seasons)
            & self.matches_df["result"].notna()
        ].copy()

        # Get test data
        test_matches = self.matches_df[
            (self.matches_df["season"] == test_season)
            & self.matches_df["result"].notna()
        ].copy()

        # Filter by matchdays if specified
        if test_matchdays is not None:
            if isinstance(test_matchdays, int):
                test_matchdays = [test_matchdays]
            test_matches = test_matches[test_matches["matchday"].isin(test_matchdays)]

        if len(test_matches) == 0:
            print("No completed test matches found!")
            return None

        print(f"Training: {train_seasons} seasons ({len(train_matches)} matches)")
        print(f"Testing: {test_season} season ({len(test_matches)} matches)")

        if test_matchdays:
            print(f"Test Matchdays: {test_matchdays}")

        # Train model
        print("\nTraining model...")
        model = self.train_model(train_matches)
        print("✓ Model trained")

        # Evaluate
        results = self.evaluate_model(model, train_matches, test_matches)

        # Show detailed predictions
        self._show_test_predictions(model, test_matches)

        return results

    def _show_test_predictions(self, model, test_df, max_show=20):
        """Show detailed predictions for test matches"""
        print("=" * 60)
        print("SAMPLE PREDICTIONS (showing first 20)")
        print("=" * 60 + "\n")

        X_test = self.create_features(test_df)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        pred_map = {0: "Away Win", 1: "Home Win", 2: "Draw"}
        actual_map = {"HOME_TEAM": "Home Win", "AWAY_TEAM": "Away Win", "DRAW": "Draw"}

        correct_count = 0

        for i, (idx, match) in enumerate(test_df.head(max_show).iterrows()):
            pred_idx = predictions[i]
            probs = probabilities[i]

            prediction = pred_map[pred_idx]
            actual = actual_map[match["result"]]
            is_correct = prediction == actual
            correct_count += is_correct

            status = "✓" if is_correct else "✗"

            print(
                f"{match['home_team']} {int(match['home_score'])}-{int(match['away_score'])} {match['away_team']}"
            )
            print(
                f"Predicted: {prediction} (H:{probs[1]*100:.0f}% D:{probs[2]*100:.0f}% A:{probs[0]*100:.0f}%)"
            )
            print(f"Actual: {actual} {status}\n")

        if len(test_df) > max_show:
            print(f"... and {len(test_df) - max_show} more matches\n")

        print(f"Shown: {correct_count}/{min(max_show, len(test_df))} correct\n")

    def run_standard_test(self):
        """Run standard test: train on 2023+2024, test on first half of 2025"""
        print("\n" + "=" * 60)
        print("STANDARD TEST: Train 2023-2024, Test 2025 (Matchdays 1-15)")
        print("=" * 60)

        return self.test_on_completed_matches(
            train_seasons=["2023", "2024"],
            test_season="2025",
            test_matchdays=list(range(1, 16)),  # First half of season
        )

    def run_cross_season_test(self):
        """Run cross-season test: train on 2023, test on 2024"""
        print("\n" + "=" * 60)
        print("CROSS-SEASON TEST: Train 2023, Test 2024")
        print("=" * 60)

        return self.test_on_completed_matches(
            train_seasons=["2023"],
            test_season="2024",
            test_matchdays=None,  # All completed matches
        )

    def run_rolling_test(self, test_season="2025", test_weeks=5):
        """Test on recent weeks using all prior data"""
        print("\n" + "=" * 60)
        print(f"ROLLING TEST: Test last {test_weeks} completed weeks of {test_season}")
        print("=" * 60)

        # Find last completed matchdays
        season_matches = self.matches_df[
            (self.matches_df["season"] == test_season)
            & self.matches_df["result"].notna()
        ]

        completed_matchdays = sorted(season_matches["matchday"].unique())

        if len(completed_matchdays) < test_weeks:
            print(f"Only {len(completed_matchdays)} completed matchdays available")
            test_matchdays = completed_matchdays
        else:
            test_matchdays = completed_matchdays[-test_weeks:]

        print(f"Testing on matchdays: {test_matchdays}\n")

        # Training: everything before test matchdays
        min_test_matchday = min(test_matchdays)

        train_matches = self.matches_df[
            (
                (self.matches_df["season"] == test_season)
                & (self.matches_df["matchday"] < min_test_matchday)
            )
            | ((self.matches_df["season"].isin(["2023", "2024"])))
        ].copy()
        train_matches = train_matches[train_matches["result"].notna()]

        test_matches = season_matches[
            season_matches["matchday"].isin(test_matchdays)
        ].copy()

        print(f"Training on {len(train_matches)} matches")
        print(f"Testing on {len(test_matches)} matches")

        # Train and evaluate
        print("\nTraining model...")
        model = self.train_model(train_matches)
        print("✓ Model trained")

        results = self.evaluate_model(model, train_matches, test_matches)
        self._show_test_predictions(model, test_matches)

        return results


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("SIMPLE LA LIGA PREDICTOR - TEST VERSION")
    print("=" * 60 + "\n")

    # Initialize and prepare data
    predictor = SimplePredictor()
    predictor.fetch_data()
    predictor.prepare_matches()
    predictor.calculate_team_stats()

    # Run standard test
    print("\nRunning standard test...")
    predictor.run_standard_test()

    # Uncomment to run other tests:
    # predictor.run_cross_season_test()
    # predictor.run_rolling_test(test_weeks=5)

    # Custom test example:
    # predictor.test_on_completed_matches(
    #     train_seasons=['2023', '2024'],
    #     test_season='2025',
    #     test_matchdays=[10, 11, 12]
    # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
