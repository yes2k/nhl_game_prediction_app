from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import polars as pl
import requests
from datetime import date, timedelta, datetime

from model import fit_model, get_table_of_predictions


templates = Jinja2Templates(directory="templates")





app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="main.html")



@app.get("/game/{date_of_pred}")
async def get_predictions(date_of_pred: str, home_team: str, away_team: str):
    # Placeholder for the actual model fitting and prediction logic

    res = await get_game_ids(date)
    if len(res) == 0:
        return {
            "error": "No games found"
        }
    else:
        print(res)
        season = res[0]["season"]

    max_date = datetime.strptime(
        date(int(date_of_pred[0:4]), int(date_of_pred[5:7]), int(date_of_pred[8:10])) - timedelta(days=1), 
        "%Y-%m-%d"
    )

    model_out = fit_model(
        "model/model.stan",
        "data/data.db",
        home_team=home_team,
        away_team=away_team,
        season=season,
        max_date=max_date
    )

    return get_table_of_predictions(model_out).as_dicts()



@app.get("/team_params/{game_id}")
async def get_team_params(team_id: int):
    pass



@app.get("/game_ids/{date}")
async def get_game_ids(date: str):
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
