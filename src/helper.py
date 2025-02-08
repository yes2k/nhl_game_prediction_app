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


def get_current_standings() -> pl.DataFrame:
    url = f"https://api-web.nhle.com/v1/standings/now"

    try:
        data = requests.get(url).json()
    except:
        print("url not found")

    out = (
        pl.DataFrame({
            "team": list(map(lambda x: x["teamAbbrev"]["default"], data["standings"])),
            "wins": list(map(lambda x: x["wins"], data["standings"])),
            "losses": list(map(lambda x: x["losses"], data["standings"])),
            "ot_losses": list(map(lambda x: x["otLosses"], data["standings"])),
            "points": list(map(lambda x: x["points"], data["standings"]))
        })
        .with_columns(
            pl.arange(1, pl.len()+1).alias("id")
        )
    )


    return out


def get_reg_scheduled_games(first_date: str, last_date: str) -> pl.DataFrame:
    # dates should be formatted as YYYY-MM-DD

    date_range = (
        pl
        .date_range(
            datetime.strptime(first_date, "%Y-%m-%d"), 
            datetime.strptime(last_date, "%Y-%m-%d"), 
            eager=True
        )
        .cast(pl.String)
        .alias("date").to_list()
    )

    out = {
        "date": [],
        "id": [],
        "home_team": [],
        "away_team": []
    }

    for d in date_range:
        url = f"https://api-web.nhle.com/v1/schedule/{d}"
        try:
            data = requests.get(url).json()
        except:
            print("url not found")
        
        # for game_date in data["gameWeek"]:
        for g in data["gameWeek"][0]["games"]:
            if g["gameType"] == 2:
                out["date"].append(d)
                out["id"].append(g["id"])
                out["away_team"].append(g["awayTeam"]["abbrev"])
                out["home_team"].append(g["homeTeam"]["abbrev"])
    
    return pl.DataFrame(out)


def get_nhl_season(date_str: str) -> str:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    if date.month > 6:  # NHL season starts in October and ends in June
        return f"{year}"
    else:
        return f"{year - 1}"




