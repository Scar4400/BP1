import requests
import pandas as pd
from config import PINNACLE_URL, LIVESCORE_URL, API_FOOTBALL_URL, HEADERS
from pytrends.request import TrendReq
from bs4 import BeautifulSoup

def fetch_data(url, headers, params):
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_pinnacle_data():
    params = {"is_have_odds": "true", "sport_id": "1"}
    return fetch_data(PINNACLE_URL, HEADERS["pinnacle"], params)

def fetch_livescore_data():
    params = {"Category": "soccer", "Query": ""}
    return fetch_data(LIVESCORE_URL, HEADERS["livescore"], params)

def fetch_api_football_data():
    params = {"page": "1"}
    return fetch_data(API_FOOTBALL_URL, HEADERS["api_football"], params)

def fetch_google_trends(teams):
    pytrends = TrendReq(hl='en-US', tz=360)
    trends_data = {}
    for team in teams:
        pytrends.build_payload([team], cat=0, timeframe='now 7-d')
        trends_data[team] = pytrends.interest_over_time().mean()[team]
    return trends_data

def fetch_news_sentiment(teams):
    sentiment_data = {}
    for team in teams:
        url = f"<https://news.google.com/rss/search?q={team}+football&hl=en-US&gl=US&ceid=US:en>"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features="xml")
        titles = soup.findAll('title')[1:6]  # Get top 5 news titles
        sentiment = sum([TextBlob(title.text).sentiment.polarity for title in titles]) / len(titles)
        sentiment_data[team] = sentiment
    return sentiment_data

def fetch_all_data():
    pinnacle_data = fetch_pinnacle_data()
    livescore_data = fetch_livescore_data()
    api_football_data = fetch_api_football_data()

    # Extract team names (this is a placeholder, adjust based on actual data structure)
    teams = set(pinnacle_data.get('teams', []) + livescore_data.get('teams', []) + api_football_data.get('teams', []))

    trends_data = fetch_google_trends(teams)
    sentiment_data = fetch_news_sentiment(teams)

    return {
        "pinnacle": pinnacle_data,
        "livescore": livescore_data,
        "api_football": api_football_data,
        "trends": trends_data,
        "sentiment": sentiment_data
    }

if __name__ == "__main__":
    data = fetch_all_data()
    for source, content in data.items():
        if content:
            print(f"{source.capitalize()} data fetched successfully")
        else:
            print(f"Failed to fetch {source} data")
