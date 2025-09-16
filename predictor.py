import requests
import pandas as pd
import os 
import json
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')

headers = {"X-Auth-Token": API_KEY}

# Get La Liga 2023 matches for training
training_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2023"
training_response = requests.get(training_url, headers=headers)
training_data = training_response.json()

# Get La Liga 2024 matches for training
testing_url = "https://api.football-data.org/v4/competitions/2014/matches?season=2024"
testing_response = requests.get(testing_url, headers=headers)
testing_data = testing_response.json()

# Structure Data
def create_matches_dataframe(data):
    matches_list = []
    for match in data['matches']:
        match_info = {'date' : match['utcDate'],
                    'matchday' : match['matchday'],
                    'home_team' : match['homeTeam']['name'],
                    'home_team_id' : match['homeTeam']['id'],
                    'away_team' : match['awayTeam']['name'],
                    'away_team_id' : match['awayTeam']['id'],
                    'home_score' : match['score']['fullTime']['home'],
                    'away_score' : match['score']['fullTime']['away'],
                    'result' : match['score']['winner']}
        matches_list.append(match_info)
    return pd.DataFrame(matches_list)
training_df = create_matches_dataframe(training_data)
testing_df = create_matches_dataframe(testing_data)

# The model needs a target to try and predict
def create_target(df):
    target_map = {
        'HOME_TEAM' : 1,
        'AWAY_TEAM' : 0,
        'DRAW' : 2
    }
    return df['result'].map(target_map)
training_target = create_target(training_df)
testing_target = create_target(testing_df)

# Features that will help model to predict
def create_features(df):
    features = pd.DataFrame()
    features['matchday'] = df['matchday']
    features['home_team_id'] = df['home_team_id']
    features['away_team_id'] = df['away_team_id']
    return features
training_features = create_features(training_df)
testing_features = create_features(testing_df)

# ML model
rf = RandomForestClassifier(n_estimators = 50, min_samples_split = 10, random_state = 1)

rf.fit(training_features, training_target)
predictions = rf.predict(testing_features)
accuracy = rf.score(testing_features, testing_target)
print(f"Accuracy : {accuracy:.2%}")