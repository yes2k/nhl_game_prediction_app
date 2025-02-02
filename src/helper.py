from datetime import date, datetime, timedelta
import polars as pl
import cmdstanpy
import sqlite3
import requests

def get_all_teams() -> pl.DataFrame:
    # TODO: figure out a better way to get all the teams for a specific season
    url = f"https://api-web.nhle.com/v1/standings/now"

    try:
        data = requests.get(url).json()
    except:
        print("url not found")

    out = (
        pl.DataFrame({
            "team": list(map(lambda x: x["teamAbbrev"]["default"], data["standings"]))
        })
        .with_columns(
            pl.arange(1, pl.len()+1).alias("id")
        )
    )

    return out


def get_game_ids(date: str):
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    
    try:
        data = requests.get(url).json()
    except Exception as e:
        print(e)
        return [{}]

    games = data['gameWeek'][0]['games']

    out = []
    for game in games:
        out.append({
            "game_id": game['id'],
            "date": date,
            "home_team": game['homeTeam']['abbrev'],
            "away_team": game['awayTeam']['abbrev'],
            "season": game['season']
        })

    return {'res': out}