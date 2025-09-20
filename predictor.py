import requests
import pandas as pd
import os 
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
import numpy as np

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
headers = {"X-Auth-Token": API_KEY}

# Get La Liga 2023 matches for training
training_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2023"
training_response = requests.get(training_url, headers=headers)
training_data = training_response.json()

# Get La Liga 2024 matches for testing
testing_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2024"
testing_response = requests.get(testing_url, headers=headers)
testing_data = testing_response.json()

# Structure Data
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

# Calculate team statistics and form
def calculate_team_stats(df):
    team_stats = {}
    
    # For each team, calculate cumulative stats up to each match
    for idx, match in df.iterrows():
        home_id = match['home_team_id']
        away_id = match['away_team_id']
        
        # Initialize team stats if first time seeing them
        if home_id not in team_stats:
            team_stats[home_id] = {
                'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                'goals_for': 0, 'goals_against': 0, 'home_wins': 0, 'home_matches': 0,
                'away_wins': 0, 'away_matches': 0, 'recent_form': []
            }
        if away_id not in team_stats:
            team_stats[away_id] = {
                'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                'goals_for': 0, 'goals_against': 0, 'home_wins': 0, 'home_matches': 0,
                'away_wins': 0, 'away_matches': 0, 'recent_form': []
            }
        
        # Store current stats before updating (for features)
        df.loc[idx, 'home_win_rate'] = team_stats[home_id]['wins'] / max(1, team_stats[home_id]['matches'])
        df.loc[idx, 'away_win_rate'] = team_stats[away_id]['wins'] / max(1, team_stats[away_id]['matches'])
        
        df.loc[idx, 'home_goals_avg'] = team_stats[home_id]['goals_for'] / max(1, team_stats[home_id]['matches'])
        df.loc[idx, 'away_goals_avg'] = team_stats[away_id]['goals_for'] / max(1, team_stats[away_id]['matches'])
        
        df.loc[idx, 'home_conceded_avg'] = team_stats[home_id]['goals_against'] / max(1, team_stats[home_id]['matches'])
        df.loc[idx, 'away_conceded_avg'] = team_stats[away_id]['goals_against'] / max(1, team_stats[away_id]['matches'])
        
        # Home advantage
        df.loc[idx, 'home_advantage'] = team_stats[home_id]['home_wins'] / max(1, team_stats[home_id]['home_matches'])
        df.loc[idx, 'away_disadvantage'] = 1 - (team_stats[away_id]['away_wins'] / max(1, team_stats[away_id]['away_matches']))
        
        # Recent form (last 5 matches)
        home_form = team_stats[home_id]['recent_form'][-5:] if team_stats[home_id]['recent_form'] else [0]
        away_form = team_stats[away_id]['recent_form'][-5:] if team_stats[away_id]['recent_form'] else [0]
        
        df.loc[idx, 'home_form'] = np.mean(home_form)
        df.loc[idx, 'away_form'] = np.mean(away_form)
        
        # Update stats after the match
        home_score = match['home_score']
        away_score = match['away_score']
        result = match['result']
        
        # Update home team stats
        team_stats[home_id]['matches'] += 1
        team_stats[home_id]['home_matches'] += 1
        team_stats[home_id]['goals_for'] += home_score
        team_stats[home_id]['goals_against'] += away_score
        
        # Update away team stats  
        team_stats[away_id]['matches'] += 1
        team_stats[away_id]['away_matches'] += 1
        team_stats[away_id]['goals_for'] += away_score
        team_stats[away_id]['goals_against'] += home_score
        
        # Update wins/draws/losses
        if result == 'HOME_TEAM':
            team_stats[home_id]['wins'] += 1
            team_stats[home_id]['home_wins'] += 1
            team_stats[away_id]['losses'] += 1
            team_stats[home_id]['recent_form'].append(3)  # 3 points for win
            team_stats[away_id]['recent_form'].append(0)  # 0 points for loss
        elif result == 'AWAY_TEAM':
            team_stats[away_id]['wins'] += 1
            team_stats[away_id]['away_wins'] += 1
            team_stats[home_id]['losses'] += 1
            team_stats[home_id]['recent_form'].append(0)
            team_stats[away_id]['recent_form'].append(3)
        else:  # DRAW
            team_stats[home_id]['draws'] += 1
            team_stats[away_id]['draws'] += 1
            team_stats[home_id]['recent_form'].append(1)  # 1 point for draw
            team_stats[away_id]['recent_form'].append(1)
    
    return df

# Calculate stats for training data
training_df = calculate_team_stats(training_df.copy())

# For testing data, we need to continue from training data stats
# Combine and calculate, then split back
combined_df = pd.concat([training_df, testing_df]).sort_values('date')
combined_df = calculate_team_stats(combined_df)
testing_df = combined_df.iloc[len(training_df):].copy()

# Create target variable
def create_target(df):
    target_map = {'HOME_TEAM': 1, 'AWAY_TEAM': 0, 'DRAW': 2}
    return df['result'].map(target_map)

training_target = create_target(training_df)
testing_target = create_target(testing_df)

# Create enhanced features
def create_features(df):
    features = pd.DataFrame()
    
    # Basic features
    features['matchday'] = df['matchday']
    features['home_team_id'] = df['home_team_id']
    features['away_team_id'] = df['away_team_id']
    
    # Performance-based features
    features['home_win_rate'] = df['home_win_rate'].fillna(0)
    features['away_win_rate'] = df['away_win_rate'].fillna(0)
    features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
    
    # Scoring features
    features['home_goals_avg'] = df['home_goals_avg'].fillna(0)
    features['away_goals_avg'] = df['away_goals_avg'].fillna(0)
    features['home_conceded_avg'] = df['home_conceded_avg'].fillna(0)
    features['away_conceded_avg'] = df['away_conceded_avg'].fillna(0)
    
    # Goal difference capabilities
    features['home_goal_diff'] = features['home_goals_avg'] - features['home_conceded_avg']
    features['away_goal_diff'] = features['away_goals_avg'] - features['away_conceded_avg']
    
    # Home advantage
    features['home_advantage'] = df['home_advantage'].fillna(0.5)
    features['away_disadvantage'] = df['away_disadvantage'].fillna(0.5)
    
    # Recent form
    features['home_form'] = df['home_form'].fillna(1)
    features['away_form'] = df['away_form'].fillna(1)
    features['form_diff'] = features['home_form'] - features['away_form']
    
    # Season timing
    features['is_early_season'] = (df['matchday'] <= 10).astype(int)
    features['is_late_season'] = (df['matchday'] >= 28).astype(int)
    
    return features

training_features = create_features(training_df)
testing_features = create_features(testing_df)

print(f"Features: {list(training_features.columns)}")

# Train model
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=1)
rf.fit(training_features, training_target)

# Make predictions
predictions = rf.predict(testing_features)
accuracy = rf.score(testing_features, testing_target)

print(f"\nAccuracy: {accuracy:.2%}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': training_features.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Show some example predictions
result_names = {0: 'Away Win', 1: 'Home Win', 2: 'Draw'}
for i in range(5):
    actual = result_names[testing_target.iloc[i]]
    predicted = result_names[predictions[i]]
    match_info = f"Team {testing_df.iloc[i]['home_team_id']} vs {testing_df.iloc[i]['away_team_id']}"
    print(f"{match_info}: Actual={actual}, Predicted={predicted}")