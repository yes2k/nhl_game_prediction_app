from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import polars as pl
import requests
from datetime import date, timedelta, datetime

import model as model
# import src.helper as helper
import helper as helper


templates = Jinja2Templates(directory="templates")

Mod = model.GamePredModel(
    "data/data.db",
    "src/model/model.stan"
)



app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="main3.html")


@app.get("/game/{date_of_pred}")
async def get_predictions(date_of_pred: str, home_team: str, away_team: str):
    season = helper.get_nhl_season(date_of_pred)

    out = Mod.get_prediction(date_of_pred, season, home_team, away_team)
    return { 
        'table_of_pred': out.pred_table.to_dicts(), 
        'home_team_win_prob': round(out.prob_home_team_win * 100, 2)
    }


@app.get("/season_projection")
async def get_season_projection():
    return Mod.get_season_prediction()


@app.get("/season_projection_plot")
async def get_season_projection_plot():
    return Mod.get_season_projection_box_plot()


@app.get("/game/{date_of_pred}/heatmap")
async def get_heatmap(date_of_pred: str, home_team: str, away_team: str):
    season = helper.get_nhl_season(date_of_pred)
    return Mod.get_prediction_heatmap_html(date_of_pred, season, home_team, away_team)


@app.get("/team_params")
async def get_team_params():
    return {
        "team_params": Mod.get_team_params().to_dicts()
    }


@app.get("/game_ids/{date}")
async def get_all_games(date: str):
    return helper.get_game_ids(date)["res"]
