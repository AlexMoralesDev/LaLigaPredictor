import requests
import pandas as pd
import os 
import json
from datetime import datetime
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = {"X-Auth-Token": API_KEY}

# File paths for tracking
PREDICTIONS_FILE = 'predictions_history.json'
README_FILE = 'README.md'

class LaLigaPredictor:
    def __init__(self):
        self.predictions_history = self.load_predictions_history()
        
    def load_predictions_history(self):
        """Load existing predictions history from JSON file"""
        if os.path.exists(PREDICTIONS_FILE):
            try:
                with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_predictions_history(self):
        """Save predictions history to JSON file"""
        with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.predictions_history, f, indent=2, ensure_ascii=False)
    
    def get_la_liga_data(self):
        """Fetch La Liga data from API"""
        print("Fetching data from API...")
        
        # Get La Liga matches for training (2023 and 2024) and current season (2025)
        training_2023_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2023"
        training_2024_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2024"
        current_2025_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2025"
        
        training_2023_response = requests.get(training_2023_url, headers=headers)
        training_2024_response = requests.get(training_2024_url, headers=headers)
        current_2025_response = requests.get(current_2025_url, headers=headers)
        
        return (training_2023_response.json(), 
                training_2024_response.json(), 
                current_2025_response.json())
    
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
            'recent_results': [],  # W=3, D=1, L=0 for last matches
            'recent_gf': [], 'recent_ga': [],
            'vs_top_teams': {'matches': 0, 'points': 0},  # vs teams in top 6
            'vs_bottom_teams': {'matches': 0, 'points': 0},  # vs teams in bottom 6
            'goal_margins': [],  # track winning/losing margins
            'clean_sheets': 0,
            'btts_against': 0,  # both teams to score against this team
            'high_scoring': 0,  # matches with 3+ total goals
            'comeback_wins': 0,
            'late_goals': 0  # goals scored after 70th minute (we'll approximate)
        })
        
        for idx, match in df.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            home_score = match['home_score']
            away_score = match['away_score']
            result = match['result']
            matchday = match['matchday']
            
            # Skip if no result (upcoming match)
            if pd.isna(result):
                # For upcoming matches, still calculate stats based on current data
                home_stats = team_stats[home_id]
                away_stats = team_stats[away_id]
                
                # Calculate current stats
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
                
                # Recent form
                recent_5 = home_stats['recent_results'][-5:] if len(home_stats['recent_results']) >= 5 else home_stats['recent_results']
                recent_3 = home_stats['recent_results'][-3:] if len(home_stats['recent_results']) >= 3 else home_stats['recent_results']
                away_recent_5 = away_stats['recent_results'][-5:] if len(away_stats['recent_results']) >= 5 else away_stats['recent_results']
                away_recent_3 = away_stats['recent_results'][-3:] if len(away_stats['recent_results']) >= 3 else away_stats['recent_results']
                
                df.loc[idx, 'home_form_5'] = np.mean(recent_5) if recent_5 else 1.0
                df.loc[idx, 'home_form_3'] = np.mean(recent_3) if recent_3 else 1.0
                df.loc[idx, 'away_form_5'] = np.mean(away_recent_5) if away_recent_5 else 1.0
                df.loc[idx, 'away_form_3'] = np.mean(away_recent_3) if away_recent_3 else 1.0
                
                # Recent scoring form
                recent_gf_5 = home_stats['recent_gf'][-5:] if len(home_stats['recent_gf']) >= 5 else home_stats['recent_gf']
                recent_ga_5 = home_stats['recent_ga'][-5:] if len(home_stats['recent_ga']) >= 5 else home_stats['recent_ga']
                away_recent_gf_5 = away_stats['recent_gf'][-5:] if len(away_stats['recent_gf']) >= 5 else away_stats['recent_gf']
                away_recent_ga_5 = away_stats['recent_ga'][-5:] if len(away_stats['recent_ga']) >= 5 else away_stats['recent_ga']
                
                df.loc[idx, 'home_recent_scoring'] = np.mean(recent_gf_5) if recent_gf_5 else 0
                df.loc[idx, 'away_recent_scoring'] = np.mean(away_recent_gf_5) if away_recent_gf_5 else 0
                df.loc[idx, 'home_recent_conceding'] = np.mean(recent_ga_5) if recent_ga_5 else 0
                df.loc[idx, 'away_recent_conceding'] = np.mean(away_recent_ga_5) if away_recent_ga_5 else 0
                
                # Performance against different quality opposition
                home_vs_top = home_stats['vs_top_teams']
                home_vs_bottom = home_stats['vs_bottom_teams']
                away_vs_top = away_stats['vs_top_teams']
                away_vs_bottom = away_stats['vs_bottom_teams']
                
                df.loc[idx, 'home_vs_top_ppg'] = home_vs_top['points'] / max(1, home_vs_top['matches'])
                df.loc[idx, 'home_vs_bottom_ppg'] = home_vs_bottom['points'] / max(1, home_vs_bottom['matches'])
                df.loc[idx, 'away_vs_top_ppg'] = away_vs_top['points'] / max(1, away_vs_top['matches'])
                df.loc[idx, 'away_vs_bottom_ppg'] = away_vs_bottom['points'] / max(1, away_vs_bottom['matches'])
                
                # Quality indicators
                df.loc[idx, 'home_clean_sheet_rate'] = home_stats['clean_sheets'] / max(1, home_stats['matches'])
                df.loc[idx, 'away_clean_sheet_rate'] = away_stats['clean_sheets'] / max(1, away_stats['matches'])
                
                df.loc[idx, 'home_btts_rate'] = home_stats['btts_against'] / max(1, home_stats['matches'])
                df.loc[idx, 'away_btts_rate'] = away_stats['btts_against'] / max(1, away_stats['matches'])
                
                # Average goal margins
                home_margins = home_stats['goal_margins'][-10:] if home_stats['goal_margins'] else [0]
                away_margins = away_stats['goal_margins'][-10:] if away_stats['goal_margins'] else [0]
                df.loc[idx, 'home_avg_margin'] = np.mean(home_margins)
                df.loc[idx, 'away_avg_margin'] = np.mean(away_margins)
                
                continue
            
            # Calculate current league position based on points per match
            current_positions = {}
            for team_id in team_stats:
                if team_stats[team_id]['matches'] > 0:
                    ppg = team_stats[team_id]['points'] / team_stats[team_id]['matches']
                    gd = team_stats[team_id]['goals_for'] - team_stats[team_id]['goals_against']
                    current_positions[team_id] = (ppg, gd, team_stats[team_id]['goals_for'])
            
            # Sort teams by points per game, then goal difference, then goals for
            sorted_teams = sorted(current_positions.items(), 
                                key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True)
            
            # Determine if teams are top 6 or bottom 6
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
            
            # Store current stats before updating
            home_stats = team_stats[home_id]
            away_stats = team_stats[away_id]
            
            # Points per game (better than win rate for draws)
            df.loc[idx, 'home_ppg'] = home_stats['points'] / max(1, home_stats['matches'])
            df.loc[idx, 'away_ppg'] = away_stats['points'] / max(1, away_stats['matches'])
            
            # Goal difference per game
            home_gd = home_stats['goals_for'] - home_stats['goals_against']
            away_gd = away_stats['goals_for'] - away_stats['goals_against']
            df.loc[idx, 'home_gd_per_game'] = home_gd / max(1, home_stats['matches'])
            df.loc[idx, 'away_gd_per_game'] = away_gd / max(1, away_stats['matches'])
            
            # Home/Away specific performance
            df.loc[idx, 'home_home_ppg'] = home_stats['home_points'] / max(1, home_stats['home_matches'])
            df.loc[idx, 'away_away_ppg'] = away_stats['away_points'] / max(1, away_stats['away_matches'])
            
            # Attack/Defense ratings (goals per game)
            df.loc[idx, 'home_attack'] = home_stats['home_gf'] / max(1, home_stats['home_matches'])
            df.loc[idx, 'away_attack'] = away_stats['away_gf'] / max(1, away_stats['away_matches'])
            df.loc[idx, 'home_defense'] = home_stats['home_ga'] / max(1, home_stats['home_matches'])
            df.loc[idx, 'away_defense'] = away_stats['away_ga'] / max(1, away_stats['away_matches'])
            
            # Recent form (last 5 and 3 matches)
            recent_5 = home_stats['recent_results'][-5:] if len(home_stats['recent_results']) >= 5 else home_stats['recent_results']
            recent_3 = home_stats['recent_results'][-3:] if len(home_stats['recent_results']) >= 3 else home_stats['recent_results']
            away_recent_5 = away_stats['recent_results'][-5:] if len(away_stats['recent_results']) >= 5 else away_stats['recent_results']
            away_recent_3 = away_stats['recent_results'][-3:] if len(away_stats['recent_results']) >= 3 else away_stats['recent_results']
            
            df.loc[idx, 'home_form_5'] = np.mean(recent_5) if recent_5 else 1.0
            df.loc[idx, 'home_form_3'] = np.mean(recent_3) if recent_3 else 1.0
            df.loc[idx, 'away_form_5'] = np.mean(away_recent_5) if away_recent_5 else 1.0
            df.loc[idx, 'away_form_3'] = np.mean(away_recent_3) if away_recent_3 else 1.0
            
            # Recent scoring form
            recent_gf_5 = home_stats['recent_gf'][-5:] if len(home_stats['recent_gf']) >= 5 else home_stats['recent_gf']
            recent_ga_5 = home_stats['recent_ga'][-5:] if len(home_stats['recent_ga']) >= 5 else home_stats['recent_ga']
            away_recent_gf_5 = away_stats['recent_gf'][-5:] if len(away_stats['recent_gf']) >= 5 else away_stats['recent_gf']
            away_recent_ga_5 = away_stats['recent_ga'][-5:] if len(away_stats['recent_ga']) >= 5 else away_stats['recent_ga']
            
            df.loc[idx, 'home_recent_scoring'] = np.mean(recent_gf_5) if recent_gf_5 else 0
            df.loc[idx, 'away_recent_scoring'] = np.mean(away_recent_gf_5) if away_recent_gf_5 else 0
            df.loc[idx, 'home_recent_conceding'] = np.mean(recent_ga_5) if recent_ga_5 else 0
            df.loc[idx, 'away_recent_conceding'] = np.mean(away_recent_ga_5) if away_recent_ga_5 else 0
            
            # Performance against different quality opposition
            home_vs_top = home_stats['vs_top_teams']
            home_vs_bottom = home_stats['vs_bottom_teams']
            away_vs_top = away_stats['vs_top_teams']
            away_vs_bottom = away_stats['vs_bottom_teams']
            
            df.loc[idx, 'home_vs_top_ppg'] = home_vs_top['points'] / max(1, home_vs_top['matches'])
            df.loc[idx, 'home_vs_bottom_ppg'] = home_vs_bottom['points'] / max(1, home_vs_bottom['matches'])
            df.loc[idx, 'away_vs_top_ppg'] = away_vs_top['points'] / max(1, away_vs_top['matches'])
            df.loc[idx, 'away_vs_bottom_ppg'] = away_vs_bottom['points'] / max(1, away_vs_bottom['matches'])
            
            # Quality indicators
            df.loc[idx, 'home_clean_sheet_rate'] = home_stats['clean_sheets'] / max(1, home_stats['matches'])
            df.loc[idx, 'away_clean_sheet_rate'] = away_stats['clean_sheets'] / max(1, away_stats['matches'])
            
            df.loc[idx, 'home_btts_rate'] = home_stats['btts_against'] / max(1, home_stats['matches'])
            df.loc[idx, 'away_btts_rate'] = away_stats['btts_against'] / max(1, away_stats['matches'])
            
            # Average goal margins
            home_margins = home_stats['goal_margins'][-10:] if home_stats['goal_margins'] else [0]
            away_margins = away_stats['goal_margins'][-10:] if away_stats['goal_margins'] else [0]
            df.loc[idx, 'home_avg_margin'] = np.mean(home_margins)
            df.loc[idx, 'away_avg_margin'] = np.mean(away_margins)
            
            # Now update stats after match
            goal_margin = home_score - away_score
            total_goals = home_score + away_score
            is_btts = home_score > 0 and away_score > 0
            is_high_scoring = total_goals >= 3
            
            # Update basic stats
            team_stats[home_id]['matches'] += 1
            team_stats[away_id]['matches'] += 1
            team_stats[home_id]['home_matches'] += 1
            team_stats[away_id]['away_matches'] += 1
            
            # Update goals
            team_stats[home_id]['goals_for'] += home_score
            team_stats[home_id]['goals_against'] += away_score
            team_stats[home_id]['home_gf'] += home_score
            team_stats[home_id]['home_ga'] += away_score
            
            team_stats[away_id]['goals_for'] += away_score
            team_stats[away_id]['goals_against'] += home_score
            team_stats[away_id]['away_gf'] += away_score
            team_stats[away_id]['away_ga'] += home_score
            
            # Update recent goals
            team_stats[home_id]['recent_gf'].append(home_score)
            team_stats[home_id]['recent_ga'].append(away_score)
            team_stats[away_id]['recent_gf'].append(away_score)
            team_stats[away_id]['recent_ga'].append(home_score)
            
            # Keep recent stats to 10 matches
            if len(team_stats[home_id]['recent_gf']) > 10:
                team_stats[home_id]['recent_gf'] = team_stats[home_id]['recent_gf'][-10:]
                team_stats[home_id]['recent_ga'] = team_stats[home_id]['recent_ga'][-10:]
            if len(team_stats[away_id]['recent_gf']) > 10:
                team_stats[away_id]['recent_gf'] = team_stats[away_id]['recent_gf'][-10:]
                team_stats[away_id]['recent_ga'] = team_stats[away_id]['recent_ga'][-10:]
            
            # Update results and points
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
            else:  # DRAW
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
            
            # Keep recent results to 10 matches
            if len(team_stats[home_id]['recent_results']) > 10:
                team_stats[home_id]['recent_results'] = team_stats[home_id]['recent_results'][-10:]
            if len(team_stats[away_id]['recent_results']) > 10:
                team_stats[away_id]['recent_results'] = team_stats[away_id]['recent_results'][-10:]
            
            # Update quality-based stats
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
            
            # Update other stats
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
        
        # Core strength indicators
        features['ppg_difference'] = df['home_ppg'].fillna(1.5) - df['away_ppg'].fillna(1.5)
        features['gd_difference'] = df['home_gd_per_game'].fillna(0) - df['away_gd_per_game'].fillna(0)
        
        # Venue-specific strength
        features['home_advantage'] = df['home_home_ppg'].fillna(1.5) - df['away_away_ppg'].fillna(1.5)
        
        # Attack vs Defense matchup
        features['attack_vs_defense'] = df['home_attack'].fillna(1) - df['away_defense'].fillna(1)
        features['away_attack_vs_home_defense'] = df['away_attack'].fillna(1) - df['home_defense'].fillna(1)
        features['overall_attacking_edge'] = features['attack_vs_defense'] - features['away_attack_vs_home_defense']
        
        # Form metrics
        features['recent_form_diff'] = df['home_form_5'].fillna(1) - df['away_form_5'].fillna(1)
        features['very_recent_form_diff'] = df['home_form_3'].fillna(1) - df['away_form_3'].fillna(1)
        
        # Scoring form
        features['recent_scoring_diff'] = df['home_recent_scoring'].fillna(0.5) - df['away_recent_scoring'].fillna(0.5)
        features['recent_defensive_diff'] = df['away_recent_conceding'].fillna(1) - df['home_recent_conceding'].fillna(1)
        
        # Quality of opposition performance
        features['quality_performance_diff'] = (
            df['home_vs_top_ppg'].fillna(1) - df['away_vs_top_ppg'].fillna(1) +
            df['home_vs_bottom_ppg'].fillna(2) - df['away_vs_bottom_ppg'].fillna(2)
        ) / 2
        
        # Defensive solidity
        features['clean_sheet_advantage'] = df['home_clean_sheet_rate'].fillna(0.2) - df['away_clean_sheet_rate'].fillna(0.2)
        
        # Goal expectation
        features['expected_home_goals'] = df['home_attack'].fillna(1) + (1 - df['away_defense'].fillna(1))
        features['expected_away_goals'] = df['away_attack'].fillna(1) + (1 - df['home_defense'].fillna(1))
        features['expected_goal_difference'] = features['expected_home_goals'] - features['expected_away_goals']
        
        # Match context
        features['matchday'] = df['matchday']
        features['is_early_season'] = (df['matchday'] <= 8).astype(int)
        features['is_crucial_period'] = ((df['matchday'] >= 30) | (df['matchday'] <= 5)).astype(int)
        
        # Combined strength indicators
        features['overall_strength_diff'] = (
            features['ppg_difference'] * 0.4 + 
            features['home_advantage'] * 0.3 + 
            features['recent_form_diff'] * 0.3
        )
        
        # Motivation/pressure
        features['home_avg_margin'] = df['home_avg_margin'].fillna(0)
        features['away_avg_margin'] = df['away_avg_margin'].fillna(0)
        features['margin_difference'] = features['home_avg_margin'] - features['away_avg_margin']
        
        # High-level indicators for tree splits
        features['home_much_stronger'] = (features['overall_strength_diff'] > 0.5).astype(int)
        features['away_much_stronger'] = (features['overall_strength_diff'] < -0.5).astype(int)
        features['evenly_matched'] = (np.abs(features['overall_strength_diff']) < 0.2).astype(int)
        
        features['home_in_form'] = (features['recent_form_diff'] > 0.5).astype(int)
        features['away_in_form'] = (features['recent_form_diff'] < -0.5).astype(int)
        
        return features
    
    def clean_predictions_history(self, current_2025_data):
        """Clean up invalid predictions and ensure proper gameweek progression"""
        print("Cleaning predictions history...")
        
        # Get all valid gameweeks from current season
        all_2025_matches = self.create_matches_dataframe(current_2025_data)
        valid_gameweeks = set()
        
        for gw in sorted(all_2025_matches['matchday'].unique()):
            gw_matches = all_2025_matches[all_2025_matches['matchday'] == gw]
            # Only consider gameweeks with reasonable match count (8-12 matches for La Liga)
            if 8 <= len(gw_matches) <= 12:
                valid_gameweeks.add(gw)
        
        print(f"Valid gameweeks found: {sorted(valid_gameweeks)}")
        
        # Remove predictions for invalid gameweeks
        invalid_predictions = []
        for gw_str in list(self.predictions_history.keys()):
            gw_int = int(gw_str)
            if gw_int not in valid_gameweeks or gw_int < 7:  # Remove anything before gameweek 7
                invalid_predictions.append(gw_str)
        
        if invalid_predictions:
            print(f"Removing invalid predictions for gameweeks: {invalid_predictions}")
            for gw_str in invalid_predictions:
                del self.predictions_history[gw_str]
            self.save_predictions_history()
        
        return valid_gameweeks
    
    def check_gameweek_readiness(self, target_gameweek, all_2025_matches, valid_gameweeks):
        """Check if we can predict a gameweek (all previous gameweeks must be completed)"""
        if target_gameweek not in valid_gameweeks:
            return False, f"Gameweek {target_gameweek} is not valid"
        
        # Check if all previous gameweeks (from 7 onwards) are completed
        previous_gameweeks = [gw for gw in valid_gameweeks if 7 <= gw < target_gameweek]
        
        for prev_gw in previous_gameweeks:
            prev_gw_matches = all_2025_matches[all_2025_matches['matchday'] == prev_gw]
            incomplete_matches = len(prev_gw_matches[prev_gw_matches['result'].isna()])
            
            if incomplete_matches > 0:
                return False, f"Cannot predict gameweek {target_gameweek}. Gameweek {prev_gw} still has {incomplete_matches} incomplete matches."
        
        return True, "Ready for prediction"
    
    def find_next_gameweek_to_predict(self, all_2025_matches, valid_gameweeks):
        """Find the next gameweek that should be predicted based on completion status"""
        predicted_gameweeks = [int(gw) for gw in self.predictions_history.keys()]
        
        # Start from gameweek 7 and find the first unpredicted gameweek that's ready
        for gw in sorted([g for g in valid_gameweeks if g >= 7]):
            if gw not in predicted_gameweeks:
                ready, message = self.check_gameweek_readiness(gw, all_2025_matches, valid_gameweeks)
                if ready:
                    return gw, message
                else:
                    print(f"Gameweek {gw}: {message}")
                    return None, message
        
        return None, "All valid gameweeks have been predicted"
    
    def update_predictions_with_results(self, current_2025_data):
        """Update previous predictions with actual results"""
        completed_matches = []
        for match in current_2025_data['matches']:
            if match['score']['winner'] is not None and match['status'] == 'FINISHED':
                completed_matches.append({
                    'matchday': match['matchday'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'],
                    'away_score': match['score']['fullTime']['away'],
                    'result': match['score']['winner'],
                    'date': match['utcDate']
                })
        
        # Update predictions history with actual results
        for completed_match in completed_matches:
            gameweek = str(completed_match['matchday'])
            
            if gameweek in self.predictions_history:
                for prediction in self.predictions_history[gameweek]['matches']:
                    if (prediction['home_team'] == completed_match['home_team'] and 
                        prediction['away_team'] == completed_match['away_team']):
                        
                        # Convert API result to our format
                        if completed_match['result'] == 'HOME_TEAM':
                            actual_result = f"{completed_match['home_team']} Win"
                        elif completed_match['result'] == 'AWAY_TEAM':
                            actual_result = f"{completed_match['away_team']} Win"
                        else:
                            actual_result = "Draw"
                        
                        prediction['actual_result'] = actual_result
                        prediction['home_score'] = completed_match['home_score']
                        prediction['away_score'] = completed_match['away_score']
                        prediction['correct'] = prediction['predicted_result'] == actual_result
        """Update previous predictions with actual results"""
        completed_matches = []
        for match in current_2025_data['matches']:
            if match['score']['winner'] is not None and match['status'] == 'FINISHED':
                completed_matches.append({
                    'matchday': match['matchday'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'],
                    'away_score': match['score']['fullTime']['away'],
                    'result': match['score']['winner'],
                    'date': match['utcDate']
                })
        
        # Update predictions history with actual results
        for completed_match in completed_matches:
            gameweek = str(completed_match['matchday'])
            
            if gameweek in self.predictions_history:
                for prediction in self.predictions_history[gameweek]['matches']:
                    if (prediction['home_team'] == completed_match['home_team'] and 
                        prediction['away_team'] == completed_match['away_team']):
                        
                        # Convert API result to our format
                        if completed_match['result'] == 'HOME_TEAM':
                            actual_result = f"{completed_match['home_team']} Win"
                        elif completed_match['result'] == 'AWAY_TEAM':
                            actual_result = f"{completed_match['away_team']} Win"
                        else:
                            actual_result = "Draw"
                        
                        prediction['actual_result'] = actual_result
                        prediction['home_score'] = completed_match['home_score']
                        prediction['away_score'] = completed_match['away_score']
                        prediction['correct'] = prediction['predicted_result'] == actual_result
    
    def calculate_accuracy(self):
        """Calculate overall accuracy from predictions history"""
        total_predictions = 0
        correct_predictions = 0
        
        for gameweek_data in self.predictions_history.values():
            for match in gameweek_data['matches']:
                if 'actual_result' in match:
                    total_predictions += 1
                    if match['correct']:
                        correct_predictions += 1
        
        if total_predictions == 0:
            return 0.0
        return (correct_predictions / total_predictions) * 100
    
    def save_predictions(self, gameweek, matches_predictions):
        """Save predictions for a specific gameweek"""
        gameweek_key = str(gameweek)
        
        # Don't overwrite if predictions already exist for this gameweek
        if gameweek_key in self.predictions_history:
            print(f"Predictions for gameweek {gameweek} already exist. Skipping...")
            return
        
        self.predictions_history[gameweek_key] = {
            'date_predicted': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'matches': matches_predictions
        }
        self.save_predictions_history()
    
    def generate_readme(self, current_gameweek, current_predictions, training_accuracy):
        """Generate README.md file with current predictions and history"""
        
        # Calculate current accuracy
        overall_accuracy = self.calculate_accuracy()
        
        readme_content = f"""# 🏆 La Liga Match Predictor

An AI-powered machine learning model that predicts La Liga match outcomes using advanced statistical analysis.

## 📊 Current Status

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model Training Accuracy:** {training_accuracy:.1f}%  
**Overall Prediction Accuracy:** {overall_accuracy:.1f}%  
**Total Matches Predicted:** {sum(len(gw['matches']) for gw in self.predictions_history.values())}  
**Total Matches with Results:** {sum(len([m for m in gw['matches'] if 'actual_result' in m]) for gw in self.predictions_history.values())}  

"""

        # Determine current vs historical gameweeks
        current_gw_int = int(current_gameweek) if current_gameweek and current_gameweek != "TBD" else None
        
        # Split predictions into current and historical
        historical_gameweeks = []
        current_gameweek_data = None
        
        for gw_str, gw_data in self.predictions_history.items():
            gw_int = int(gw_str)
            
            # Check if this gameweek has any upcoming matches
            has_upcoming = any('actual_result' not in match for match in gw_data['matches'])
            
            if gw_int == current_gw_int and has_upcoming:
                # This is the current gameweek with upcoming matches
                current_gameweek_data = (gw_str, gw_data)
            else:
                # This is a historical gameweek (either older or all matches completed)
                historical_gameweeks.append((gw_str, gw_data))
        
        # Sort historical gameweeks
        historical_gameweeks.sort(key=lambda x: int(x[0]))
        
        # Add current gameweek predictions
        if current_gameweek_data or current_predictions:
            if current_gameweek_data:
                display_gw = current_gameweek_data[0]
                display_predictions = current_gameweek_data[1]['matches']
            else:
                display_gw = current_gameweek
                display_predictions = current_predictions
            
            readme_content += f"## 🔮 Current Gameweek Predictions\n\n### Gameweek {display_gw}\n\n"
            
            if display_predictions:
                for pred in display_predictions:
                    # Check if match is completed or upcoming
                    if 'actual_result' in pred:
                        status_icon = "✅" if pred.get('correct', False) else "❌"
                        status_text = f"**Result:** {pred['actual_result']} ({pred.get('home_score', '?')}-{pred.get('away_score', '?')}) - {'✅ CORRECT' if pred.get('correct', False) else '❌ WRONG'}"
                    else:
                        status_icon = "⏳"
                        status_text = "**Status:** Awaiting result"
                    
                    readme_content += f"""{status_icon} **{pred['home_team']} vs {pred['away_team']}**  
📅 {pred['date']}  
🎯 **Prediction:** {pred['predicted_result']}  
{status_text}  
📊 Probabilities: {pred['home_team']} {pred['home_prob']:.1f}% | Draw {pred['draw_prob']:.1f}% | {pred['away_team']} {pred['away_prob']:.1f}%  

"""
            else:
                readme_content += "No predictions available for current gameweek.\n\n"
        else:
            readme_content += "## 🔮 Current Gameweek Predictions\n\nNo upcoming matches found for prediction.\n\n"
        
        # Add prediction history (only for completed/historical gameweeks)
        readme_content += "## 📈 Prediction History\n\n"
        
        if historical_gameweeks:
            for gameweek_str, gw_data in historical_gameweeks:
                gameweek = int(gameweek_str)
                readme_content += f"### Gameweek {gameweek}\n"
                readme_content += f"*Predicted on: {gw_data['date_predicted']}*\n\n"
                
                # Calculate gameweek accuracy
                gw_total = len(gw_data['matches'])
                gw_correct = sum(1 for match in gw_data['matches'] if match.get('correct', False))
                gw_completed = sum(1 for match in gw_data['matches'] if 'actual_result' in match)
                
                if gw_completed > 0:
                    gw_accuracy = (gw_correct / gw_completed) * 100
                    readme_content += f"**Final Accuracy: {gw_accuracy:.1f}% ({gw_correct}/{gw_completed} correct)**\n\n"
                elif gw_total > 0:
                    readme_content += f"**Status: {gw_total} matches predicted, awaiting results**\n\n"
                else:
                    readme_content += "**Status: No matches**\n\n"
                
                # List matches
                for match in gw_data['matches']:
                    if 'actual_result' in match:
                        status_icon = "✅" if match.get('correct', False) else "❌"
                        result_text = f"⚽ **Final Result:** {match['actual_result']} ({match.get('home_score', '?')}-{match.get('away_score', '?')})"
                    else:
                        status_icon = "⏳"
                        result_text = "⏳ **Status:** Result pending"
                    
                    readme_content += f"{status_icon} **{match['home_team']} vs {match['away_team']}**  \n"
                    readme_content += f"🎯 Predicted: {match['predicted_result']}  \n"
                    readme_content += f"{result_text}  \n"
                    readme_content += f"📊 Confidence: {match['home_prob']:.1f}% | {match['draw_prob']:.1f}% | {match['away_prob']:.1f}%\n\n"
                
                readme_content += "---\n\n"
        else:
            readme_content += "No historical predictions available yet. Predictions will appear here after gameweeks are completed.\n\n"
        
        # Add model information
        readme_content += """## 🤖 Model Information

### Features Used
- Points per game difference
- Goal difference per game
- Home/Away venue-specific performance
- Attack vs Defense matchup analysis
- Recent form (last 3 and 5 matches)
- Performance against top/bottom teams
- Clean sheet rates and defensive metrics
- Expected goals calculations

### Algorithm
- **Random Forest Classifier** with 300 trees
- Trained on 2+ seasons of La Liga data
- Features engineered for maximum predictive power
- Handles class imbalance with balanced weights

## 📋 How to Use

1. Clone this repository
2. Set up your Football-Data.org API key in `.env` file
3. Run `python predictor.py` to generate new predictions
4. Check this README for the latest predictions and results

## 📊 Accuracy Breakdown

The model's accuracy is tracked across different result types:
- **Home Wins**: Historically strong performance
- **Away Wins**: Moderate accuracy  
- **Draws**: Most challenging to predict (as expected)

---

*Predictions are for entertainment purposes only. Past performance does not guarantee future results.*
"""
        
        # Write README file
        with open(README_FILE, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"README.md updated successfully!")
    
    def run_prediction(self, force_gameweek=None, start_from_gameweek=7):
        """Main method to run the prediction process
        
        Args:
            force_gameweek (int): Force prediction of a specific gameweek (optional)
            start_from_gameweek (int): Minimum gameweek to consider (default: 7)
        """
        # Get data
        training_2023_data, training_2024_data, current_2025_data = self.get_la_liga_data()
        
        # Clean predictions history first
        valid_gameweeks = self.clean_predictions_history(current_2025_data)
        
        # Create dataframes
        training_2023_df = self.create_matches_dataframe(training_2023_data)
        training_2024_df = self.create_matches_dataframe(training_2024_data)
        current_2025_df = self.create_matches_dataframe(current_2025_data)
        
        # Update predictions with actual results
        self.update_predictions_with_results(current_2025_data)
        
        # Filter only finished matches for training
        training_2023_df = training_2023_df[training_2023_df['result'].notna()].copy()
        training_2024_df = training_2024_df[training_2024_df['result'].notna()].copy()
        
        # Separate completed and upcoming matches in 2025
        completed_2025_df = current_2025_df[current_2025_df['result'].notna()].copy()
        upcoming_2025_df = current_2025_df[current_2025_df['result'].isna()].copy()
        
        print(f"Training matches 2023: {len(training_2023_df)}")
        print(f"Training matches 2024: {len(training_2024_df)}")
        print(f"Completed matches 2025: {len(completed_2025_df)}")
        print(f"Upcoming matches 2025: {len(upcoming_2025_df)}")
        
        # Find the gameweek to predict
        current_gameweek = None
        gameweek_matches = pd.DataFrame()
        
        # Get all 2025 matches
        all_2025_matches = current_2025_df.copy()
        
        if len(all_2025_matches) > 0:
            if force_gameweek is not None:
                # Check if forced gameweek is valid and ready
                ready, message = self.check_gameweek_readiness(force_gameweek, all_2025_matches, valid_gameweeks)
                if ready:
                    current_gameweek = force_gameweek
                    print(f"FORCING prediction of gameweek: {current_gameweek}")
                    
                    # Remove from predictions history if it exists (to re-predict)
                    if str(force_gameweek) in self.predictions_history:
                        print(f"Removing existing predictions for gameweek {force_gameweek}")
                        del self.predictions_history[str(force_gameweek)]
                        self.save_predictions_history()
                else:
                    print(f"Cannot force gameweek {force_gameweek}: {message}")
                    return 0.0
            else:
                # Find next gameweek automatically
                current_gameweek, message = self.find_next_gameweek_to_predict(all_2025_matches, valid_gameweeks)
                if current_gameweek is None:
                    print(f"No gameweek ready for prediction: {message}")
                    # Still generate README with current data
                    self.generate_readme("TBD", [], 0.0)
                    return self.calculate_accuracy()
                else:
                    print(f"Selected gameweek {current_gameweek} for prediction: {message}")
            
            # Get ALL matches for the selected gameweek
            gameweek_matches = all_2025_matches[all_2025_matches['matchday'] == current_gameweek].copy()
            print(f"Total matches in gameweek {current_gameweek}: {len(gameweek_matches)}")
            
            # Separate for display purposes
            completed_in_gw = gameweek_matches[gameweek_matches['result'].notna()]
            upcoming_in_gw = gameweek_matches[gameweek_matches['result'].isna()]
            print(f"  - Completed: {len(completed_in_gw)}")
            print(f"  - Upcoming: {len(upcoming_in_gw)}")
            
            # Show the matches to verify
            print(f"\nMatches in gameweek {current_gameweek}:")
            for _, match in gameweek_matches.iterrows():
                status = "COMPLETED" if pd.notna(match['result']) else "UPCOMING"
                print(f"  {match['home_team_name']} vs {match['away_team_name']} - {status}")
        else:
            print("No 2025 matches found!")
            self.generate_readme("TBD", [], 0.0)
            return 0.0
        
        current_predictions = []
        training_accuracy = 0.0
        
        if current_gameweek is not None and len(gameweek_matches) > 0:
            # For training, use all completed matches up to (but not including) current gameweek
            training_matches = pd.concat([
                training_2023_df, 
                training_2024_df, 
                current_2025_df[
                    (current_2025_df['matchday'] < current_gameweek) & 
                    (current_2025_df['result'].notna())
                ]
            ]).sort_values('date').reset_index(drop=True)
            
            print(f"Training on matches up to gameweek {current_gameweek-1}: {len(training_matches)} matches")
            
            # Combine training data with current gameweek for stats calculation
            all_data_with_current = pd.concat([training_matches, gameweek_matches]).sort_values('date').reset_index(drop=True)
            
            # Calculate stats
            print("Calculating enhanced predictive statistics...")
            all_data_with_stats = self.calculate_predictive_stats(all_data_with_current.copy())
            
            # Split back into training and prediction data
            training_data_with_stats = all_data_with_stats.iloc[:len(training_matches)].copy()
            prediction_data_with_stats = all_data_with_stats.iloc[len(training_matches):].copy()
            
            # Create features and targets
            training_features = self.create_focused_features(training_data_with_stats)
            training_target = self.create_target(training_data_with_stats)
            prediction_features = self.create_focused_features(prediction_data_with_stats)
            
            print(f"Training features shape: {training_features.shape}")
            print(f"Prediction features shape: {prediction_features.shape}")
            
            # Train the model
            print("Training optimized Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                class_weight='balanced'
            )
            
            rf.fit(training_features, training_target)
            
            # Make predictions
            predictions = rf.predict(prediction_features)
            prediction_probabilities = rf.predict_proba(prediction_features)
            
            # Calculate training accuracy
            training_accuracy = rf.score(training_features, training_target) * 100
            
            print(f"\n{'='*60}")
            print(f"PREDICTIONS FOR GAMEWEEK {current_gameweek}")
            print(f"{'='*60}")
            
            # Prepare predictions for saving
            matches_predictions = []
            
            for i, (idx, match) in enumerate(prediction_data_with_stats.iterrows()):
                home_team = match['home_team_name']
                away_team = match['away_team_name']
                
                # Create result names
                if predictions[i] == 0:  # Away win
                    predicted_result = f"{away_team} Win"
                elif predictions[i] == 1:  # Home win
                    predicted_result = f"{home_team} Win"
                else:  # Draw
                    predicted_result = "Draw"
                
                # Get probabilities
                home_prob = prediction_probabilities[i][1] * 100
                away_prob = prediction_probabilities[i][0] * 100
                draw_prob = prediction_probabilities[i][2] * 100
                
                match_date = match['date'].strftime('%Y-%m-%d %H:%M')
                
                # Check if match is already completed
                actual_result = None
                home_score = None
                away_score = None
                is_correct = None
                
                if pd.notna(match['result']):
                    # Match is completed, get actual result
                    if match['result'] == 'HOME_TEAM':
                        actual_result = f"{home_team} Win"
                    elif match['result'] == 'AWAY_TEAM':
                        actual_result = f"{away_team} Win"
                    else:
                        actual_result = "Draw"
                    
                    home_score = int(match['home_score'])
                    away_score = int(match['away_score'])
                    is_correct = predicted_result == actual_result
                
                # Create prediction record
                prediction_record = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_result': predicted_result,
                    'home_prob': round(home_prob, 1),
                    'away_prob': round(away_prob, 1),
                    'draw_prob': round(draw_prob, 1),
                    'date': match_date
                }
                
                # Add actual results if available
                if actual_result is not None:
                    prediction_record['actual_result'] = actual_result
                    prediction_record['home_score'] = home_score
                    prediction_record['away_score'] = away_score
                    prediction_record['correct'] = is_correct
                
                matches_predictions.append(prediction_record)
                current_predictions.append(prediction_record)
                
                # Print prediction
                status = "✅" if is_correct else "❌" if is_correct is False else "⏳"
                print(f"\n{status} {match_date}")
                print(f"{home_team} vs {away_team}")
                print(f"Prediction: {predicted_result}")
                
                if actual_result is not None:
                    print(f"Actual: {actual_result} ({home_score}-{away_score})")
                    print(f"Result: {'CORRECT' if is_correct else 'WRONG'}")
                
                print(f"Probabilities: {home_team} {home_prob:.1f}% | Draw {draw_prob:.1f}% | {away_team} {away_prob:.1f}%")
            
            # Save predictions
            self.save_predictions(current_gameweek, matches_predictions)
            
            # Feature importance
            importance_df = pd.DataFrame({
                'feature': training_features.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n{'='*60}")
            print("TOP 10 MOST IMPORTANT FEATURES:")
            print(f"{'='*60}")
            for i, row in importance_df.head(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
            
            print(f"\nModel training accuracy: {training_accuracy:.2%}")
            
            # Calculate gameweek accuracy if any matches are completed
            completed_predictions = [p for p in matches_predictions if 'actual_result' in p]
            if completed_predictions:
                gw_correct = sum(1 for p in completed_predictions if p['correct'])
                gw_accuracy = (gw_correct / len(completed_predictions)) * 100
                print(f"Gameweek {current_gameweek} accuracy: {gw_accuracy:.1f}% ({gw_correct}/{len(completed_predictions)})")
        
        # Generate README
        self.generate_readme(current_gameweek or "TBD", current_predictions, training_accuracy)
        
        # Final summary
        overall_accuracy = self.calculate_accuracy()
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Overall prediction accuracy: {overall_accuracy:.1f}%")
        print(f"✅ Generated predictions for gameweek {current_gameweek}")
        print(f"✅ README.md updated with current status")
        print(f"✅ Predictions history maintained")
        
        return overall_accuracy

if __name__ == "__main__":
    import sys
    
    predictor = LaLigaPredictor()
    
    # Check if user wants to force a specific gameweek or clean history
    if len(sys.argv) > 1:
        if sys.argv[1] == "clean":
            print("Cleaning predictions history...")
            predictor.predictions_history = {}
            predictor.save_predictions_history()
            print("Predictions history cleared!")
        else:
            try:
                force_gw = int(sys.argv[1])
                print(f"Forcing prediction of gameweek {force_gw}")
                predictor.run_prediction(force_gameweek=force_gw)
            except ValueError:
                print("Invalid gameweek number provided. Use 'clean' to clear history or a number for specific gameweek.")
                predictor.run_prediction()
    else:
        # Default: start looking from gameweek 7
        predictor.run_prediction(start_from_gameweek=7)