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

# Get La Liga matches
training_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2023"
training_response = requests.get(training_url, headers=headers)
training_data = training_response.json()

testing_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2024"
testing_response = requests.get(testing_url, headers=headers)
testing_data = testing_response.json()

# Create matches dataframe
def create_matches_dataframe(data):
    matches_list = []
    for match in data['matches']:
        if match['score']['winner'] == None:
            continue
            
        match_info = {
            'date': pd.to_datetime(match['utcDate']),
            'matchday': match['matchday'],
            'home_team_id': match['homeTeam']['id'],
            'away_team_id': match['awayTeam']['id'],
            'home_team_name': match['homeTeam']['name'],
            'away_team_name': match['awayTeam']['name'],
            'home_score': match['score']['fullTime']['home'],
            'away_score': match['score']['fullTime']['away'],
            'result': match['score']['winner']
        }
        matches_list.append(match_info)
    
    return pd.DataFrame(matches_list).sort_values('date')

training_df = create_matches_dataframe(training_data)
testing_df = create_matches_dataframe(testing_data)

print(f"Training matches: {len(training_df)}")
print(f"Testing matches: {len(testing_df)}")

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

# Calculate stats
print("Calculating enhanced predictive statistics...")
training_df = calculate_predictive_stats(training_df.copy())

# For testing, continue from training stats
combined_df = pd.concat([training_df, testing_df]).sort_values('date')
combined_df = calculate_predictive_stats(combined_df)
testing_df = combined_df.iloc[len(training_df):].copy()

# Create target
def create_target(df):
    target_map = {'HOME_TEAM': 1, 'AWAY_TEAM': 0, 'DRAW': 2}
    return df['result'].map(target_map)

training_target = create_target(training_df)
testing_target = create_target(testing_df)

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

training_features = create_focused_features(training_df)
testing_features = create_focused_features(testing_df)

print(f"Focused features: {len(training_features.columns)} features")

# Optimized Random Forest
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

print("\nTraining optimized Random Forest...")
rf.fit(training_features, training_target)

# Predictions and evaluation
predictions = rf.predict(testing_features)
accuracy = rf.score(testing_features, testing_target)

print(f"Random Forest Accuracy: {accuracy:.2%}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': training_features.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Prediction analysis
result_names = {0: 'Away Win', 1: 'Home Win', 2: 'Draw'}
print(f"\nSample Predictions:")

for i in range(min(8, len(testing_df))):
    actual = result_names[testing_target.iloc[i]]
    predicted = result_names[predictions[i]]
    home_team = testing_df.iloc[i]['home_team_name']
    away_team = testing_df.iloc[i]['away_team_name']
    matchday = testing_df.iloc[i]['matchday']
    
    print(f"MD{matchday}: {home_team} vs {away_team}")
    print(f"  Predicted: {predicted}, Actual: {actual} {'✓' if actual == predicted else '✗'}")

# Performance by result type
print(f"\nAccuracy by result type:")
for result_code, result_name in result_names.items():
    mask = testing_target == result_code
    if mask.sum() > 0:
        acc = (predictions[mask] == testing_target[mask]).mean()
        count = mask.sum()
        print(f"{result_name}: {acc:.2%} ({count} matches)")

print(f"\nAccuracy: {accuracy:.2%}")