import requests
import pandas as pd
import os 
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')

headers = {"X-Auth-Token": API_KEY}

# Get La Liga matches
url = "https://api.football-data.org/v4/competitions/2014/matches"
response = requests.get(url, headers=headers)
data = response.json()
matches_list = []

# Structure Data
for match in data['matches']:
    match_info = {'date' : match['utcDate'],
                  'matchday' : match['matchday'],
                  'home_team' : match['homeTeam']['name'],
                  'home_team_id' : match['homeTeam']['id'],
                  'away_team' : match['awayTeam']['name'],
                  'away_team_id' : match['awayTeam']['id'],
                  'home_score' : match['score']['fullTime']['home'],
                  'away_score' : match['score']['fullTime']['away'],
                  'winner' : match['score']['winner']}
    matches_list.append(match_info)
df = pd.DataFrame(matches_list)