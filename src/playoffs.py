import model
import helper

import json
from datetime import date
import polars as pl
import re


class PlayoffSim:
    def __init__(self, path_to_bracket: str):
        with open(path_to_bracket, 'r') as file:
            self.nhl_playoff_bracket = json.load(file)
    
    def model_bracket(self):
        Mod = model.GamePredModel(
            "data/data.db",
            "src/model/model.stan"
        )

        # round 1 eastern conference
        for k, v in self.nhl_playoff_bracket["round_1"]["eastern_conference"]["matchups"].items():
            home_team = v["home"]
            away_team = v["away"]
            date_of_pred = date.today().strftime("%Y-%m-%d")  # Get today's date in YYYY-MM-DD format
            season = helper.get_nhl_season(date_of_pred)
            out = Mod.get_playoff_prediction(date_of_pred, season, home_team, away_team)
            winning_team = out.filter(pl.col("prob") == pl.col("prob").max())["map"][0]
            v["winner"] = winning_team[0:3]
            v["games"] = re.findall("\\d", winning_team)[0]
        
        print(self.nhl_playoff_bracket)

