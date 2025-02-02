from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import polars as pl
import requests
from datetime import date, timedelta, datetime

import model as model
import helper as helper

templates = Jinja2Templates(directory="templates")

Mod = model.GamePredModel(
    "data/data.db",
    "src/model/model.stan"
)



app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="main.html")



@app.get("/game/{date_of_pred}")
async def get_predictions(date_of_pred: str, home_team: str, away_team: str):
    res = helper.get_game_ids(date_of_pred)
    if len(res) == 0:
        return {
            "error": "No games found"
        }
    else:
        season = str(res['res'][0]["season"])[:4]

    out = Mod.get_prediction(date_of_pred, season, home_team, away_team)

    print(out)

    return { 
        'table_of_pred': out.pred_table.to_dicts(), 
        'team_params': out.team_params.to_dicts(),
        'home_team_win_prob': out.prob_home_team_win * 100
    }



@app.get("/team_params/{game_id}")
async def get_team_params(team_id: int):
    pass



@app.get("/game_ids/{date}")
async def get_all_games(date: str):
    return helper.get_game_ids(date)["res"]
