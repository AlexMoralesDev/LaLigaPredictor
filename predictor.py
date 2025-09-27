import requests
import pandas as pd
import os 
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = {"X-Auth-Token": API_KEY}

# Get La Liga matches for training (2023 and 2024) and current season (2025)
training_2023_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2023"
training_2024_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2024"
current_2025_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2025"

print("Fetching data from API...")
training_2023_response = requests.get(training_2023_url, headers=headers)
training_2024_response = requests.get(training_2024_url, headers=headers)
current_2025_response = requests.get(current_2025_url, headers=headers)

training_2023_data = training_2023_response.json()
training_2024_data = training_2024_response.json()
current_2025_data = current_2025_response.json()

# Create matches dataframe
def create_matches_dataframe(data):
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

training_2023_df = create_matches_dataframe(training_2023_data)
training_2024_df = create_matches_dataframe(training_2024_data)
current_2025_df = create_matches_dataframe(current_2025_data)

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

# Find the next gameweek to predict
if len(upcoming_2025_df) > 0:
    next_gameweek = upcoming_2025_df['matchday'].min()
    print(f"Next gameweek to predict: {next_gameweek}")
    
    # Get matches for the next gameweek
    next_gameweek_matches = upcoming_2025_df[upcoming_2025_df['matchday'] == next_gameweek].copy()
    print(f"Matches in gameweek {next_gameweek}: {len(next_gameweek_matches)}")
else:
    print("No upcoming matches found!")
    next_gameweek = None

# Enhanced statistics with focus on predictive power
def calculate_predictive_stats(df):
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
    
    # First pass: calculate league position for each matchday to identify top/bottom teams
    league_positions = {}
    
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

# Create target
def create_target(df):
    target_map = {'HOME_TEAM': 1, 'AWAY_TEAM': 0, 'DRAW': 2}
    return df['result'].map(target_map)

# Create focused feature set
def create_focused_features(df):
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

print("\nCalculating enhanced predictive statistics...")

# Combine all training data (2023 + 2024 + completed 2025 matches)
all_training_data = pd.concat([training_2023_df, training_2024_df, completed_2025_df]).sort_values('date').reset_index(drop=True)
print(f"Total training matches: {len(all_training_data)}")

# Calculate stats for all data including upcoming matches
if next_gameweek is not None:
    # Combine all data for stats calculation
    all_data_with_upcoming = pd.concat([all_training_data, next_gameweek_matches]).sort_values('date').reset_index(drop=True)
    
    # Calculate stats
    all_data_with_stats = calculate_predictive_stats(all_data_with_upcoming.copy())
    
    # Split back into training and prediction data
    training_data_with_stats = all_data_with_stats.iloc[:len(all_training_data)].copy()
    prediction_data_with_stats = all_data_with_stats.iloc[len(all_training_data):].copy()
    
    # Create features and targets
    training_features = create_focused_features(training_data_with_stats)
    training_target = create_target(training_data_with_stats)
    prediction_features = create_focused_features(prediction_data_with_stats)
    
    print(f"Training features shape: {training_features.shape}")
    print(f"Prediction features shape: {prediction_features.shape}")
    
    # Train the model
    print("\nTraining optimized Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,          # More trees for stability
        max_depth=12,              # Deeper trees for complex patterns
        min_samples_split=8,       # Prevent overfitting
        min_samples_leaf=3,        # Ensure meaningful leaves
        max_features='sqrt',       # Feature sampling
        bootstrap=True,
        random_state=42,
        class_weight='balanced'    # Handle class imbalance
    )
    
    rf.fit(training_features, training_target)
    
    # Make predictions for next gameweek
    predictions = rf.predict(prediction_features)
    prediction_probabilities = rf.predict_proba(prediction_features)
    
    print(f"\n{'='*60}")
    print(f"PREDICTIONS FOR GAMEWEEK {next_gameweek}")
    print(f"{'='*60}")
    
    for i, (idx, match) in enumerate(prediction_data_with_stats.iterrows()):
        home_team = match['home_team_name']
        away_team = match['away_team_name']
        
        # Create result names with actual team names
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
        
        # Get match date
        match_date = match['date'].strftime('%Y-%m-%d %H:%M')
        
        print(f"\n{match_date}")
        print(f"{home_team} vs {away_team}")
        print(f"Prediction: {predicted_result}")
        print(f"Probabilities: {home_team} {home_prob:.1f}% | Draw {draw_prob:.1f}% | {away_team} {away_prob:.1f}%")
        
        # Show key stats
        home_ppg = prediction_data_with_stats.iloc[i]['home_ppg']
        away_ppg = prediction_data_with_stats.iloc[i]['away_ppg']
        home_form = prediction_data_with_stats.iloc[i]['home_form_5']
        away_form = prediction_data_with_stats.iloc[i]['away_form_5']
        
        print(f"Stats: {home_team} PPG: {home_ppg:.2f}, Form: {home_form:.2f} | {away_team} PPG: {away_ppg:.2f}, Form: {away_form:.2f}")
    
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
    
    # Model performance on training data (for reference)
    training_accuracy = rf.score(training_features, training_target)
    print(f"\nModel training accuracy: {training_accuracy:.2%}")
    
    # Performance by result type on training data
    training_predictions = rf.predict(training_features)
    result_names = {0: 'Away Win', 1: 'Home Win', 2: 'Draw'}
    print(f"\nTraining accuracy by result type:")
    for result_code, result_name in result_names.items():
        mask = training_target == result_code
        if mask.sum() > 0:
            acc = (training_predictions[mask] == training_target[mask]).mean()
            count = mask.sum()
            print(f"{result_name}: {acc:.2%} ({count} matches)")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Trained on {len(all_training_data)} matches from 2023-2025")
    print(f"✅ Model accuracy on training data: {training_accuracy:.2%}")
    print(f"✅ Generated predictions for {len(next_gameweek_matches)} matches in gameweek {next_gameweek}")
    print(f"✅ Features used: {len(training_features.columns)} predictive features")
    
else:
    print("No upcoming matches found to predict!")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")