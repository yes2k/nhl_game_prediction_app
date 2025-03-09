import polars as pl
import sqlite3
import requests
from datetime import date, timedelta, datetime
import sys
import time
from argparse import ArgumentParser
import os
import json

# import src.model as model
# import src.helper as helper 

import model as model
import helper as helper 

def get_reg_goals(date: str):
    def get_reg_goals_single_game(x, d):
        out = {}
        out["date"] = d
        out["id"] = x["id"]

        out["away_team"] = x["awayTeam"]["abbrev"]
        out["home_team"] = x["homeTeam"]["abbrev"]

        reg_goals = list(filter(lambda x: x["period"] <= 3, x["goals"]))
        all_goals = x["goals"]

        if len(reg_goals) != 0:
            out["home_goals"] = reg_goals[-1]["homeScore"]
            out["away_goals"] = reg_goals[-1]["awayScore"]
        else:
            out["home_goals"] = 0
            out["away_goals"] = 0
        
        if all_goals[-1]["homeScore"] > all_goals[-1]["awayScore"]:
            out["winning_team"] = out["home_team"]
        else:
            out["winning_team"] = out["away_team"]


        return out

    

    url = f"https://api-web.nhle.com/v1/score/{date}"
    try:
        data = requests.get(url).json()
    except:
        print("url not found")



    out = []
    for game in data["games"]:
        if game["gameType"] == 2 and 'goals' in game.keys():
            out.append(get_reg_goals_single_game(game, date))
    
    return out


def build_database(start_date: str, path_to_db: str) -> None:
    """
    start_date should be formatted as YYYY-MM-DD    
    """
    start_date = date(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
    end_date = date.today() - timedelta(days = 1)
    date_range = pl.date_range(start_date, end_date, eager=True).cast(pl.String).alias("date").to_list()
    
    # Getting all the regular season goals for t \in [start_date, today - (1 day)]
    out = []
    for d in date_range:
        out.append(pl.DataFrame(get_reg_goals(d)))

    df = pl.concat(out, how = "diagonal")

    df.write_database(
        table_name = "goal_data",
        connection = f"sqlite:///{path_to_db}/data.db",
        if_table_exists="replace"
    )

    # Getting the team parameters and pred goals based on model
    # pred_goal_data = []
    # team_params = []
    mod = model.GamePredModel(f"{path_to_db}/data.db", "src/model/model.stan")
    # for d in date_range[1:]:
    #     for g in helper.get_game_ids(d)["res"]:
    #         if str(g["game_id"])[4:6] == '02':
    #             print(g["game_id"])
    #             out = mod.get_prediction(d, helper.get_nhl_season(d), g["home_team"], g["away_team"])
    #             team_params.append(
    #                 out.team_params
    #                 .select(["50%", "team", "type"])
    #                 .with_columns(
    #                     pl.lit(d).alias("date_of_pred")
    #                 )
    #                 .rename({"50%":"param"})
    #             )
    
    # getting predictions for the current day, and next day
    pred_goal_data = []
    date_range2 = [date.today().strftime("%Y-%m-%d"), (date.today() + timedelta(days = 1)).strftime("%Y-%m-%d")]
    for d in date_range2:
        for g in helper.get_game_ids(d)["res"]:
            if str(g["game_id"])[4:6] == '02':
                d2 = (date(int(d[0:4]), int(d[5:7]), int(d[8:10])) - timedelta(days=1)).strftime("%Y-%m-%d")
                out = mod.get_prediction(d2, helper.get_nhl_season(d), g["home_team"], g["away_team"])
                pred_goal_data.append(
                    out.pred_table.with_columns(
                        pl.lit(d).alias("date_of_game"),
                        pl.lit(g["game_id"]).alias("game_id"),
                        pl.lit(g["home_team"]).alias("home_team"),
                        pl.lit(g["away_team"]).alias("away_team"),
                        pl.lit(out.prob_home_team_win).alias("prob_home_team_win")
                    )
                )

    (
        pl.concat(pred_goal_data, how = "vertical_relaxed")
        .write_database(
            table_name = "pred_goal_data",
            connection = f"sqlite:///{path_to_db}/data.db"
        )
    )

    # Getting latest team parameters
    (
        mod.get_team_params()
        .write_database(
            table_name = "team_params",
            connection = f"sqlite:///{path_to_db}/data.db"
        )
    )

    # Getting the season predictions
    season_proj = mod.get_season_prediction()
    with open('data/seasons_proj.json', 'w') as f:
        json.dump(season_proj, f)
    

def update_database(path_to_db: str) -> None:
    con = sqlite3.connect(f"{path_to_db}/data.db")
    cur = con.cursor()
    max_date = cur.execute("SELECT MAX(date) FROM goal_data").fetchall()[0][0]
    con.close()

    max_date = datetime.strptime(max_date, "%Y-%m-%d")

    date_range = (
        pl
        .date_range(
            max_date + timedelta(days = 1), 
            date.today() - timedelta(days = 1), 
            eager=True
        )
        .cast(pl.String)
        .alias("date").to_list()
    )

    out = []
    for d in date_range:
        print(d)
        out.append(pl.DataFrame(get_reg_goals(d)))
    
    if len(out) == 0:
        print("No new data to update")
        return

    df = pl.concat(out, how = "diagonal")
    df.write_database(
        table_name = "goal_data",
        connection = f"sqlite:///{path_to_db}/data.db",
        if_table_exists="append"
    )

    # team_params = []
    mod = model.GamePredModel(f"{path_to_db}/data.db", "src/model/model.stan")
    # for d in date_range:
    #     for g in helper.get_game_ids(d)["res"]:
    #         if str(g["game_id"])[4:6] == '02':
    #             print(g["game_id"])
    #             max_date = date(int(d[0:4]), int(d[5:7]), int(d[8:10])) - timedelta(days = 1)
    #             out = mod.get_prediction(max_date, helper.get_nhl_season(d), g["home_team"], g["away_team"])
    #             team_params.append(
    #                 out.team_params
    #                 .select(["50%", "team", "type"])
    #                 .with_columns(
    #                     pl.lit(d).alias("date_of_pred")
    #                 )
    #                 .rename({"50%":"param"})
    #             )
    

    # getting predictions for the current day, and next day
    pred_goal_data = []
    date_range2 = [date.today().strftime("%Y-%m-%d"), (date.today() + timedelta(days = 1)).strftime("%Y-%m-%d")]
    for d in date_range2:
        for g in helper.get_game_ids(d)["res"]:
            if str(g["game_id"])[4:6] == '02':
                d2 = (date(int(d[0:4]), int(d[5:7]), int(d[8:10])) - timedelta(days=1)).strftime("%Y-%m-%d")
                out = mod.get_prediction(d2, helper.get_nhl_season(d), g["home_team"], g["away_team"])
                pred_goal_data.append(
                    out.pred_table.with_columns(
                        pl.lit(d).alias("date_of_game"),
                        pl.lit(g["game_id"]).alias("game_id"),
                        pl.lit(g["home_team"]).alias("home_team"),
                        pl.lit(g["away_team"]).alias("away_team"),
                        pl.lit(out.prob_home_team_win).alias("prob_home_team_win")
                    )
                )

    (
        pl.concat(pred_goal_data, how = "vertical_relaxed")
        .write_database(
            table_name = "pred_goal_data",
            connection = f"sqlite:///{path_to_db}/data.db",
            if_table_exists="replace"
        )
    )

    # Getting latest team parameters
    (
        mod.get_team_params(date_range2[0], helper.get_nhl_season(date_range2[0]))
        .write_database(
            table_name = "team_params",
            connection = f"sqlite:///{path_to_db}/data.db",
            if_table_exists="replace"
        )
    )

    # Getting the season predictions
    season_proj = mod.get_season_prediction()
    with open('data/seasons_proj.json', 'w') as f:
        json.dump(season_proj, f)


if __name__  == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--type")
    parser.add_argument("-p", "--pathtodb")
    parser.add_argument("-s", "--startdate")

    args = parser.parse_args()

    match args.type:
        case "rebuild": 
            if args.pathtodb is None:
                ValueError("pathtodb arg is empty")
            if args.startdate is None:
                ValueError("startdate arg is empty")
            build_database(args.startdate, args.pathtodb)
        case "update": 
            if args.pathtodb is None:
                ValueError("pathtodb arg is empty")
            update_database(args.pathtodb)
        case _: ValueError("Incorrect argument for type, takes one of 'rebuild' or 'update'")

