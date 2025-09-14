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
print(data['matches'])