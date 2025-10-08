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
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')

# Validate environment variables
if not API_KEY:
    print("ERROR: FOOTBALL_API_KEY not found in .env file")
    exit(1)

headers = {"X-Auth-Token": API_KEY}

class LaLigaPredictorDev:
    def __init__(self):
        self.dev_results = {
            'model_name': '',
            'training_accuracy': 0,
            'cross_val_scores': [],
            'predictions': [],
            'feature_importance': {},
            'confusion_matrix': None
        }
        
    def get_la_liga_data(self):
        """Fetch La Liga data from API with retry logic"""
        print("Fetching data from API...")
        
        urls = {
            '2023': "https://api.football-data.org/v4/competitions/2014/matches?season=2023",
            '2024': "https://api.football-data.org/v4/competitions/2014/matches?season=2024",
            '2025': "https://api.football-data.org/v4/competitions/2014/matches?season=2025"
        }
        
        responses = {}
        
        for season, url in urls.items():
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    print(f"  Fetching {season} season data (attempt {attempt + 1}/{max_retries})...")
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
                    if attempt < max_retries - 1:
                        print(f"  Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"\n  Failed to connect after {max_retries} attempts.")
                        exit(1)
                        
                except Exception as e:
                    print(f"  Unexpected error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise
        
        return responses['2023'], responses['2024'], responses['2025']
    
    def create_matches_dataframe(self, data):
        """Create DataFrame from API response"""
        matches_list = []
        for match in data['matches']:
            match_info = {
                'date': pd.to_datetime(match['utcDate']),
                'matchday': match['matchday'],
                'home_team_id': match['homeTeam']['id'],
                'away_team_id': match['awayTeam']['id'],
                'home_team_name': match['homeTeam']['name'],
                'away_team_name': match['awayTeam']['name'],
                'home_score': match['score']['fullTime']['home'],
                'away_score': match['score']['fullTime']['away'],
                'result': match['score']['winner'],
                'status': match['status']
            }
            matches_list.append(match_info)
        
        return pd.DataFrame(matches_list).sort_values('date')
    
    def calculate_predictive_stats(self, df):
        """Calculate enhanced predictive statistics"""
        team_stats = defaultdict(lambda: {
            'matches': 0, 'points': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0,
            'home_matches': 0, 'home_points': 0, 'home_gf': 0, 'home_ga': 0,
            'away_matches': 0, 'away_points': 0, 'away_gf': 0, 'away_ga': 0,
            'recent_results': [],
            'recent_gf': [], 'recent_ga': [],
            'vs_top_teams': {'matches': 0, 'points': 0},
            'vs_bottom_teams': {'matches': 0, 'points': 0},
            'goal_margins': [],
            'clean_sheets': 0,
            'btts_against': 0,
            'high_scoring': 0,
            'comeback_wins': 0,
            'late_goals': 0
        })
        
        for idx, match in df.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            home_score = match['home_score']
            away_score = match['away_score']
            result = match['result']
            
            if pd.isna(result):
                home_stats = team_stats[home_id]
                away_stats = team_stats[away_id]
                
                df.loc[idx, 'home_ppg'] = home_stats['points'] / max(1, home_stats['matches'])
                df.loc[idx, 'away_ppg'] = away_stats['points'] / max(1, away_stats['matches'])
                
                home_gd = home_stats['goals_for'] - home_stats['goals_against']
                away_gd = away_stats['goals_for'] - away_stats['goals_against']
                df.loc[idx, 'home_gd_per_game'] = home_gd / max(1, home_stats['matches'])
                df.loc[idx, 'away_gd_per_game'] = away_gd / max(1, away_stats['matches'])
                
                df.loc[idx, 'home_home_ppg'] = home_stats['home_points'] / max(1, home_stats['home_matches'])
                df.loc[idx, 'away_away_ppg'] = away_stats['away_points'] / max(1, away_stats['away_matches'])
                
                df.loc[idx, 'home_attack'] = home_stats['home_gf'] / max(1, home_stats['home_matches'])
                df.loc[idx, 'away_attack'] = away_stats['away_gf'] / max(1, away_stats['away_matches'])
                df.loc[idx, 'home_defense'] = home_stats['home_ga'] / max(1, home_stats['home_matches'])
                df.loc[idx, 'away_defense'] = away_stats['away_ga'] / max(1, away_stats['away_matches'])
                
                recent_5 = home_stats['recent_results'][-5:] if len(home_stats['recent_results']) >= 5 else home_stats['recent_results']
                recent_3 = home_stats['recent_results'][-3:] if len(home_stats['recent_results']) >= 3 else home_stats['recent_results']
                away_recent_5 = away_stats['recent_results'][-5:] if len(away_stats['recent_results']) >= 5 else away_stats['recent_results']
                away_recent_3 = away_stats['recent_results'][-3:] if len(away_stats['recent_results']) >= 3 else away_stats['recent_results']
                
                df.loc[idx, 'home_form_5'] = np.mean(recent_5) if recent_5 else 1.0
                df.loc[idx, 'home_form_3'] = np.mean(recent_3) if recent_3 else 1.0
                df.loc[idx, 'away_form_5'] = np.mean(away_recent_5) if away_recent_5 else 1.0
                df.loc[idx, 'away_form_3'] = np.mean(away_recent_3) if away_recent_3 else 1.0
                
                recent_gf_5 = home_stats['recent_gf'][-5:] if len(home_stats['recent_gf']) >= 5 else home_stats['recent_gf']
                recent_ga_5 = home_stats['recent_ga'][-5:] if len(home_stats['recent_ga']) >= 5 else home_stats['recent_ga']
                away_recent_gf_5 = away_stats['recent_gf'][-5:] if len(away_stats['recent_gf']) >= 5 else away_stats['recent_gf']
                away_recent_ga_5 = away_stats['recent_ga'][-5:] if len(away_stats['recent_ga']) >= 5 else away_stats['recent_ga']
                
                df.loc[idx, 'home_recent_scoring'] = np.mean(recent_gf_5) if recent_gf_5 else 0
                df.loc[idx, 'away_recent_scoring'] = np.mean(away_recent_gf_5) if away_recent_gf_5 else 0
                df.loc[idx, 'home_recent_conceding'] = np.mean(recent_ga_5) if recent_ga_5 else 0
                df.loc[idx, 'away_recent_conceding'] = np.mean(away_recent_ga_5) if away_recent_ga_5 else 0
                
                home_vs_top = home_stats['vs_top_teams']
                home_vs_bottom = home_stats['vs_bottom_teams']
                away_vs_top = away_stats['vs_top_teams']
                away_vs_bottom = away_stats['vs_bottom_teams']
                
                df.loc[idx, 'home_vs_top_ppg'] = home_vs_top['points'] / max(1, home_vs_top['matches'])
                df.loc[idx, 'home_vs_bottom_ppg'] = home_vs_bottom['points'] / max(1, home_vs_bottom['matches'])
                df.loc[idx, 'away_vs_top_ppg'] = away_vs_top['points'] / max(1, away_vs_top['matches'])
                df.loc[idx, 'away_vs_bottom_ppg'] = away_vs_bottom['points'] / max(1, away_vs_bottom['matches'])
                
                df.loc[idx, 'home_clean_sheet_rate'] = home_stats['clean_sheets'] / max(1, home_stats['matches'])
                df.loc[idx, 'away_clean_sheet_rate'] = away_stats['clean_sheets'] / max(1, away_stats['matches'])
                
                df.loc[idx, 'home_btts_rate'] = home_stats['btts_against'] / max(1, home_stats['matches'])
                df.loc[idx, 'away_btts_rate'] = away_stats['btts_against'] / max(1, away_stats['matches'])
                
                home_margins = home_stats['goal_margins'][-10:] if home_stats['goal_margins'] else [0]
                away_margins = away_stats['goal_margins'][-10:] if away_stats['goal_margins'] else [0]
                df.loc[idx, 'home_avg_margin'] = np.mean(home_margins)
                df.loc[idx, 'away_avg_margin'] = np.mean(away_margins)
                
                continue
            
            current_positions = {}
            for team_id in team_stats:
                if team_stats[team_id]['matches'] > 0:
                    ppg = team_stats[team_id]['points'] / team_stats[team_id]['matches']
                    gd = team_stats[team_id]['goals_for'] - team_stats[team_id]['goals_against']
                    current_positions[team_id] = (ppg, gd, team_stats[team_id]['goals_for'])
            
            sorted_teams = sorted(current_positions.items(), 
                                key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True)
            
            total_teams = len(sorted_teams)
            top_threshold = min(6, max(1, total_teams // 3))
            bottom_threshold = max(total_teams - 6, total_teams * 2 // 3)
            
            home_is_top = away_is_top = home_is_bottom = away_is_bottom = False
            
            for pos, (team_id, _) in enumerate(sorted_teams):
                if team_id == home_id:
                    home_is_top = pos < top_threshold
                    home_is_bottom = pos >= bottom_threshold
                elif team_id == away_id:
                    away_is_top = pos < top_threshold
                    away_is_bottom = pos >= bottom_threshold
            
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            df.loc[idx, 'home_ppg'] = home_stats['points'] / max(1, home_stats['matches'])
            df.loc[idx, 'away_ppg'] = away_stats['points'] / max(1, away_stats['matches'])
            
            home_gd = home_stats['goals_for'] - home_stats['goals_against']
            away_gd = away_stats['goals_for'] - away_stats['goals_against']
            df.loc[idx, 'home_gd_per_game'] = home_gd / max(1, home_stats['matches'])
            df.loc[idx, 'away_gd_per_game'] = away_gd / max(1, away_stats['matches'])
            
            df.loc[idx, 'home_home_ppg'] = home_stats['home_points'] / max(1, home_stats['home_matches'])
            df.loc[idx, 'away_away_ppg'] = away_stats['away_points'] / max(1, away_stats['away_matches'])
            
            df.loc[idx, 'home_attack'] = home_stats['home_gf'] / max(1, home_stats['home_matches'])
            df.loc[idx, 'away_attack'] = away_stats['away_gf'] / max(1, away_stats['away_matches'])
            df.loc[idx, 'home_defense'] = home_stats['home_ga'] / max(1, home_stats['home_matches'])
            df.loc[idx, 'away_defense'] = away_stats['away_ga'] / max(1, away_stats['away_matches'])
            
            recent_5 = home_stats['recent_results'][-5:] if len(home_stats['recent_results']) >= 5 else home_stats['recent_results']
            recent_3 = home_stats['recent_results'][-3:] if len(home_stats['recent_results']) >= 3 else home_stats['recent_results']
            away_recent_5 = away_stats['recent_results'][-5:] if len(away_stats['recent_results']) >= 5 else away_stats['recent_results']
            away_recent_3 = away_stats['recent_results'][-3:] if len(away_stats['recent_results']) >= 3 else away_stats['recent_results']
            
            df.loc[idx, 'home_form_5'] = np.mean(recent_5) if recent_5 else 1.0
            df.loc[idx, 'home_form_3'] = np.mean(recent_3) if recent_3 else 1.0
            df.loc[idx, 'away_form_5'] = np.mean(away_recent_5) if away_recent_5 else 1.0
            df.loc[idx, 'away_form_3'] = np.mean(away_recent_3) if away_recent_3 else 1.0
            
            recent_gf_5 = home_stats['recent_gf'][-5:] if len(home_stats['recent_gf']) >= 5 else home_stats['recent_gf']
            recent_ga_5 = home_stats['recent_ga'][-5:] if len(home_stats['recent_ga']) >= 5 else home_stats['recent_ga']
            away_recent_gf_5 = away_stats['recent_gf'][-5:] if len(away_stats['recent_gf']) >= 5 else away_stats['recent_gf']
            away_recent_ga_5 = away_stats['recent_ga'][-5:] if len(away_stats['recent_ga']) >= 5 else away_stats['recent_ga']
            
            df.loc[idx, 'home_recent_scoring'] = np.mean(recent_gf_5) if recent_gf_5 else 0
            df.loc[idx, 'away_recent_scoring'] = np.mean(away_recent_gf_5) if away_recent_gf_5 else 0
            df.loc[idx, 'home_recent_conceding'] = np.mean(recent_ga_5) if recent_ga_5 else 0
            df.loc[idx, 'away_recent_conceding'] = np.mean(away_recent_ga_5) if away_recent_ga_5 else 0
            
            home_vs_top = home_stats['vs_top_teams']
            home_vs_bottom = home_stats['vs_bottom_teams']
            away_vs_top = away_stats['vs_top_teams']
            away_vs_bottom = away_stats['vs_bottom_teams']
            
            df.loc[idx, 'home_vs_top_ppg'] = home_vs_top['points'] / max(1, home_vs_top['matches'])
            df.loc[idx, 'home_vs_bottom_ppg'] = home_vs_bottom['points'] / max(1, home_vs_bottom['matches'])
            df.loc[idx, 'away_vs_top_ppg'] = away_vs_top['points'] / max(1, away_vs_top['matches'])
            df.loc[idx, 'away_vs_bottom_ppg'] = away_vs_bottom['points'] / max(1, away_vs_bottom['matches'])
            
            df.loc[idx, 'home_clean_sheet_rate'] = home_stats['clean_sheets'] / max(1, home_stats['matches'])
            df.loc[idx, 'away_clean_sheet_rate'] = away_stats['clean_sheets'] / max(1, away_stats['matches'])
            
            df.loc[idx, 'home_btts_rate'] = home_stats['btts_against'] / max(1, home_stats['matches'])
            df.loc[idx, 'away_btts_rate'] = away_stats['btts_against'] / max(1, away_stats['matches'])
            
            home_margins = home_stats['goal_margins'][-10:] if home_stats['goal_margins'] else [0]
            away_margins = away_stats['goal_margins'][-10:] if away_stats['goal_margins'] else [0]
            df.loc[idx, 'home_avg_margin'] = np.mean(home_margins)
            df.loc[idx, 'away_avg_margin'] = np.mean(away_margins)
            
            goal_margin = home_score - away_score
            total_goals = home_score + away_score
            is_btts = home_score > 0 and away_score > 0
            is_high_scoring = total_goals >= 3
            
            team_stats[home_id]['matches'] += 1
            team_stats[away_id]['matches'] += 1
            team_stats[home_id]['home_matches'] += 1
            team_stats[away_id]['away_matches'] += 1
            
            team_stats[home_id]['goals_for'] += home_score
            team_stats[home_id]['goals_against'] += away_score
            team_stats[home_id]['home_gf'] += home_score
            team_stats[home_id]['home_ga'] += away_score
            
            team_stats[away_id]['goals_for'] += away_score
            team_stats[away_id]['goals_against'] += home_score
            team_stats[away_id]['away_gf'] += away_score
            team_stats[away_id]['away_ga'] += home_score
            
            team_stats[home_id]['recent_gf'].append(home_score)
            team_stats[home_id]['recent_ga'].append(away_score)
            team_stats[away_id]['recent_gf'].append(away_score)
            team_stats[away_id]['recent_ga'].append(home_score)
            
            if len(team_stats[home_id]['recent_gf']) > 10:
                team_stats[home_id]['recent_gf'] = team_stats[home_id]['recent_gf'][-10:]
                team_stats[home_id]['recent_ga'] = team_stats[home_id]['recent_ga'][-10:]
            if len(team_stats[away_id]['recent_gf']) > 10:
                team_stats[away_id]['recent_gf'] = team_stats[away_id]['recent_gf'][-10:]
                team_stats[away_id]['recent_ga'] = team_stats[away_id]['recent_ga'][-10:]
            
            if result == 'HOME_TEAM':
                team_stats[home_id]['wins'] += 1
                team_stats[home_id]['points'] += 3
                team_stats[home_id]['home_points'] += 3
                team_stats[away_id]['losses'] += 1
                team_stats[home_id]['recent_results'].append(3)
                team_stats[away_id]['recent_results'].append(0)
                team_stats[home_id]['goal_margins'].append(goal_margin)
                team_stats[away_id]['goal_margins'].append(-goal_margin)
            elif result == 'AWAY_TEAM':
                team_stats[away_id]['wins'] += 1
                team_stats[away_id]['points'] += 3
                team_stats[away_id]['away_points'] += 3
                team_stats[home_id]['losses'] += 1
                team_stats[home_id]['recent_results'].append(0)
                team_stats[away_id]['recent_results'].append(3)
                team_stats[home_id]['goal_margins'].append(goal_margin)
                team_stats[away_id]['goal_margins'].append(-goal_margin)
            else:
                team_stats[home_id]['draws'] += 1
                team_stats[away_id]['draws'] += 1
                team_stats[home_id]['points'] += 1
                team_stats[away_id]['points'] += 1
                team_stats[home_id]['home_points'] += 1
                team_stats[away_id]['away_points'] += 1
                team_stats[home_id]['recent_results'].append(1)
                team_stats[away_id]['recent_results'].append(1)
                team_stats[home_id]['goal_margins'].append(0)
                team_stats[away_id]['goal_margins'].append(0)
            
            if len(team_stats[home_id]['recent_results']) > 10:
                team_stats[home_id]['recent_results'] = team_stats[home_id]['recent_results'][-10:]
            if len(team_stats[away_id]['recent_results']) > 10:
                team_stats[away_id]['recent_results'] = team_stats[away_id]['recent_results'][-10:]
            
            if away_is_top:
                team_stats[home_id]['vs_top_teams']['matches'] += 1
                if result == 'HOME_TEAM':
                    team_stats[home_id]['vs_top_teams']['points'] += 3
                elif result == 'DRAW':
                    team_stats[home_id]['vs_top_teams']['points'] += 1
                    
            if home_is_top:
                team_stats[away_id]['vs_top_teams']['matches'] += 1
                if result == 'AWAY_TEAM':
                    team_stats[away_id]['vs_top_teams']['points'] += 3
                elif result == 'DRAW':
                    team_stats[away_id]['vs_top_teams']['points'] += 1
                    
            if away_is_bottom:
                team_stats[home_id]['vs_bottom_teams']['matches'] += 1
                if result == 'HOME_TEAM':
                    team_stats[home_id]['vs_bottom_teams']['points'] += 3
                elif result == 'DRAW':
                    team_stats[home_id]['vs_bottom_teams']['points'] += 1
                    
            if home_is_bottom:
                team_stats[away_id]['vs_bottom_teams']['matches'] += 1
                if result == 'AWAY_TEAM':
                    team_stats[away_id]['vs_bottom_teams']['points'] += 3
                elif result == 'DRAW':
                    team_stats[away_id]['vs_bottom_teams']['points'] += 1
            
            if away_score == 0:
                team_stats[home_id]['clean_sheets'] += 1
            if home_score == 0:
                team_stats[away_id]['clean_sheets'] += 1
                
            if is_btts:
                team_stats[home_id]['btts_against'] += 1
                team_stats[away_id]['btts_against'] += 1
                
            if is_high_scoring:
                team_stats[home_id]['high_scoring'] += 1
                team_stats[away_id]['high_scoring'] += 1
        
        return df
    
    def create_target(self, df):
        """Create target variable"""
        target_map = {'HOME_TEAM': 1, 'AWAY_TEAM': 0, 'DRAW': 2}
        return df['result'].map(target_map)
    
    def create_focused_features(self, df):
        """Create focused feature set"""
        features = pd.DataFrame()
        
        features['ppg_difference'] = df['home_ppg'].fillna(1.5) - df['away_ppg'].fillna(1.5)
        features['gd_difference'] = df['home_gd_per_game'].fillna(0) - df['away_gd_per_game'].fillna(0)
        features['home_advantage'] = df['home_home_ppg'].fillna(1.5) - df['away_away_ppg'].fillna(1.5)
        features['attack_vs_defense'] = df['home_attack'].fillna(1) - df['away_defense'].fillna(1)
        features['away_attack_vs_home_defense'] = df['away_attack'].fillna(1) - df['home_defense'].fillna(1)
        features['overall_attacking_edge'] = features['attack_vs_defense'] - features['away_attack_vs_home_defense']
        features['recent_form_diff'] = df['home_form_5'].fillna(1) - df['away_form_5'].fillna(1)
        features['very_recent_form_diff'] = df['home_form_3'].fillna(1) - df['away_form_3'].fillna(1)
        features['recent_scoring_diff'] = df['home_recent_scoring'].fillna(0.5) - df['away_recent_scoring'].fillna(0.5)
        features['recent_defensive_diff'] = df['away_recent_conceding'].fillna(1) - df['home_recent_conceding'].fillna(1)
        features['quality_performance_diff'] = (
            df['home_vs_top_ppg'].fillna(1) - df['away_vs_top_ppg'].fillna(1) +
            df['home_vs_bottom_ppg'].fillna(2) - df['away_vs_bottom_ppg'].fillna(2)
        ) / 2
        features['clean_sheet_advantage'] = df['home_clean_sheet_rate'].fillna(0.2) - df['away_clean_sheet_rate'].fillna(0.2)
        features['expected_home_goals'] = df['home_attack'].fillna(1) + (1 - df['away_defense'].fillna(1))
        features['expected_away_goals'] = df['away_attack'].fillna(1) + (1 - df['home_defense'].fillna(1))
        features['expected_goal_difference'] = features['expected_home_goals'] - features['expected_away_goals']
        features['matchday'] = df['matchday']
        features['is_early_season'] = (df['matchday'] <= 8).astype(int)
        features['is_crucial_period'] = ((df['matchday'] >= 30) | (df['matchday'] <= 5)).astype(int)
        features['overall_strength_diff'] = (
            features['ppg_difference'] * 0.4 + 
            features['home_advantage'] * 0.3 + 
            features['recent_form_diff'] * 0.3
        )
        features['home_avg_margin'] = df['home_avg_margin'].fillna(0)
        features['away_avg_margin'] = df['away_avg_margin'].fillna(0)
        features['margin_difference'] = features['home_avg_margin'] - features['away_avg_margin']
        features['home_much_stronger'] = (features['overall_strength_diff'] > 0.5).astype(int)
        features['away_much_stronger'] = (features['overall_strength_diff'] < -0.5).astype(int)
        features['evenly_matched'] = (np.abs(features['overall_strength_diff']) < 0.2).astype(int)
        features['home_in_form'] = (features['recent_form_diff'] > 0.5).astype(int)
        features['away_in_form'] = (features['recent_form_diff'] < -0.5).astype(int)
        
        return features
    
    def evaluate_model(self, model, X, y, model_name="Model"):
        """Evaluate model with cross-validation"""
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}\n")
        
        # Training accuracy
        train_score = model.score(X, y) * 100
        print(f"Training Accuracy: {train_score:.2f}%")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-Validation Scores: {[f'{score*100:.2f}%' for score in cv_scores]}")
        print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            self.dev_results['feature_importance'] = feature_importance.to_dict('records')
        
        self.dev_results['model_name'] = model_name
        self.dev_results['training_accuracy'] = train_score
        self.dev_results['cross_val_scores'] = cv_scores.tolist()
        
        return cv_scores.mean() * 100
    
    def save_dev_results(self, filename='dev_results.json'):
        """Save development results to JSON file"""
        output_file = f"dev_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.dev_results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    def test_model_configuration(self, model_config_name, model, test_gameweek=None):
        """Test a model configuration without updating production"""
        print("\n" + "="*60)
        print(f"LA LIGA PREDICTOR - DEVELOPMENT MODE")
        print(f"Testing: {model_config_name}")
        print("="*60 + "\n")
        
        training_2023_data, training_2024_data, current_2025_data = self.get_la_liga_data()
        
        print("\nCreating dataframes...")
        training_2023_df = self.create_matches_dataframe(training_2023_data)
        training_2024_df = self.create_matches_dataframe(training_2024_data)
        current_2025_df = self.create_matches_dataframe(current_2025_data)
        
        # Find current gameweek or use specified test gameweek
        all_2025_matches = current_2025_df.copy()
        current_gameweek = test_gameweek
        
        if current_gameweek is None:
            print("\nSearching for current gameweek...")
            for gw in sorted(all_2025_matches['matchday'].unique()):
                if gw < 7:
                    continue
                gw_matches = all_2025_matches[all_2025_matches['matchday'] == gw]
                incomplete = len(gw_matches[gw_matches['result'].isna()])
                
                if incomplete > 0:
                    current_gameweek = gw
                    print(f"✓ Found gameweek {gw} with {incomplete} incomplete matches")
                    break
        else:
            print(f"\nUsing specified test gameweek: {current_gameweek}")
        
        if current_gameweek is None:
            print("✗ No gameweek to predict")
            return
        
        # Get matches for current gameweek
        gameweek_matches = all_2025_matches[all_2025_matches['matchday'] == current_gameweek].copy()
        
        # Prepare training data
        print("\nPreparing training data...")
        training_2023_df = training_2023_df[training_2023_df['result'].notna()].copy()
        training_2024_df = training_2024_df[training_2024_df['result'].notna()].copy()
        
        training_matches = pd.concat([
            training_2023_df, 
            training_2024_df, 
            current_2025_df[
                (current_2025_df['matchday'] < current_gameweek) & 
                (current_2025_df['result'].notna())
            ]
        ]).sort_values('date').reset_index(drop=True)
        
        print(f"✓ Training on {len(training_matches)} completed matches")
        
        # Calculate stats
        print("\nCalculating team statistics...")
        all_data = pd.concat([training_matches, gameweek_matches]).sort_values('date').reset_index(drop=True)
        all_data_with_stats = self.calculate_predictive_stats(all_data.copy())
        
        training_data = all_data_with_stats.iloc[:len(training_matches)].copy()
        prediction_data = all_data_with_stats.iloc[len(training_matches):].copy()
        
        # Create features
        print("Creating features...")
        training_features = self.create_focused_features(training_data)
        training_target = self.create_target(training_data)
        prediction_features = self.create_focused_features(prediction_data)
        
        # Train and evaluate model
        print(f"\nTraining {model_config_name}...")
        model.fit(training_features, training_target)
        
        # Evaluate with cross-validation
        cv_accuracy = self.evaluate_model(model, training_features, training_target, model_config_name)
        
        # Make predictions
        print(f"\n{'='*60}")
        print(f"GAMEWEEK {current_gameweek} PREDICTIONS")
        print(f"{'='*60}\n")
        
        predictions = model.predict(prediction_features)
        probabilities = model.predict_proba(prediction_features)
        
        correct_predictions = 0
        total_completed = 0
        matches_predictions = []
        
        for i, (idx, match) in enumerate(prediction_data.iterrows()):
            home_team = match['home_team_name']
            away_team = match['away_team_name']
            
            if predictions[i] == 0:
                predicted_result = f"{away_team} Win"
                predicted_class = 'AWAY_TEAM'
            elif predictions[i] == 1:
                predicted_result = f"{home_team} Win"
                predicted_class = 'HOME_TEAM'
            else:
                predicted_result = "Draw"
                predicted_class = 'DRAW'
            
            home_prob = probabilities[i][1] * 100
            away_prob = probabilities[i][0] * 100
            draw_prob = probabilities[i][2] * 100
            
            prediction_record = {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_result': predicted_result,
                'home_prob': round(home_prob, 1),
                'away_prob': round(away_prob, 1),
                'draw_prob': round(draw_prob, 1),
                'date': match['date'].isoformat()
            }
            
            # Check if match is completed
            status_indicator = ""
            if pd.notna(match['result']):
                if match['result'] == 'HOME_TEAM':
                    actual_result = f"{home_team} Win"
                elif match['result'] == 'AWAY_TEAM':
                    actual_result = f"{away_team} Win"
                else:
                    actual_result = "Draw"
                
                prediction_record['actual_result'] = actual_result
                prediction_record['home_score'] = int(match['home_score'])
                prediction_record['away_score'] = int(match['away_score'])
                
                is_correct = predicted_class == match['result']
                prediction_record['correct'] = is_correct
                
                if is_correct:
                    correct_predictions += 1
                    status_indicator = " ✓ CORRECT"
                else:
                    status_indicator = " ✗ WRONG"
                
                total_completed += 1
            
            matches_predictions.append(prediction_record)
            
            print(f"{home_team} vs {away_team}")
            print(f"Prediction: {predicted_result}{status_indicator}")
            print(f"Confidence: {home_team} {home_prob:.1f}% | Draw {draw_prob:.1f}% | {away_team} {away_prob:.1f}%")
            
            if pd.notna(match['result']):
                print(f"Actual: {actual_result} ({int(match['home_score'])}-{int(match['away_score'])})")
            
            print()
        
        self.dev_results['predictions'] = matches_predictions
        
        # Calculate prediction accuracy on completed matches
        print(f"\n{'='*60}")
        print(f"MODEL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {model_config_name}")
        print(f"Training Accuracy: {self.dev_results['training_accuracy']:.2f}%")
        print(f"Cross-Validation Accuracy: {cv_accuracy:.2f}%")
        
        if total_completed > 0:
            prediction_accuracy = (correct_predictions / total_completed) * 100
            print(f"Prediction Accuracy on Completed Matches: {prediction_accuracy:.2f}% ({correct_predictions}/{total_completed})")
            self.dev_results['prediction_accuracy'] = prediction_accuracy
        else:
            print(f"No completed matches to evaluate prediction accuracy")
        
        print(f"{'='*60}\n")
        
        # Save results
        self.save_dev_results()
        
        return self.dev_results

if __name__ == "__main__":
    try:
        predictor = LaLigaPredictorDev()
        
        # Example: Test different model configurations
        # You can uncomment and modify these to test different models
        
        # Current production model
        print("\n🔬 TESTING PRODUCTION MODEL CONFIGURATION")
        production_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        predictor.test_model_configuration("Production Model (Current)", production_model)
        
        # Example: Test a new model with different hyperparameters
        # print("\n🔬 TESTING NEW MODEL CONFIGURATION")
        # experimental_model = RandomForestClassifier(
        #     n_estimators=500,
        #     max_depth=15,
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     max_features='sqrt',
        #     bootstrap=True,
        #     random_state=42,
        #     class_weight='balanced'
        # )
        # predictor.test_model_configuration("Experimental Model v1", experimental_model)
        
        # Example: Test with Gradient Boosting
        # from sklearn.ensemble import GradientBoostingClassifier
        # gb_model = GradientBoostingClassifier(
        #     n_estimators=200,
        #     learning_rate=0.1,
        #     max_depth=5,
        #     random_state=42
        # )
        # predictor.test_model_configuration("Gradient Boosting Model", gb_model)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
