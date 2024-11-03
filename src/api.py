from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import polars as pl
import requests
from datetime import date, timedelta, datetime

from model import fit_model, get_prediction


templates = Jinja2Templates(directory="templates")


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

    return out







app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="main.html")



@app.get("/game/{date_of_pred}")
async def get_predictions(date_of_pred: str, home_team: str, away_team: str):
    out = get_prediction(date_of_pred, home_team, away_team)

    return { 
        'table_of_pred': out['table_of_pred'].to_dicts(), 
        'team_params': out['team_params'].to_dicts()
    }



@app.get("/team_params/{game_id}")
async def get_team_params(team_id: int):
    pass



@app.get("/game_ids/{date}")
async def get_all_games(date: str):
    return get_game_ids(date)
