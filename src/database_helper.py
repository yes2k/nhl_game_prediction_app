import polars as pl
import sqlite3
import requests
from datetime import date, timedelta
import sys
import time
from argparse import ArgumentParser
import os



def get_reg_goals(date: str):
    def get_reg_goals_single_game(x, d):
        out = {}
        out["date"] = d
        out["id"] = x["id"]

        out["away_team"] = x["awayTeam"]["abbrev"]
        out["home_team"] = x["homeTeam"]["abbrev"]

        reg_goals = list(filter(lambda x: x["period"] <= 3, x["goals"]))

        if len(reg_goals) != 0:
            out["home_goals"] = reg_goals[-1]["homeScore"]
            out["away_goals"] = reg_goals[-1]["awayScore"]
        else:
            out["home_goals"] = 0
            out["away_goals"] = 0

        return out


    url = f"https://api-web.nhle.com/v1/score/{date}"
    try:
        data = requests.get(url).json()
    except:
        print("url not found")
    
    # print(url)

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
    
    # try: 
    # conn = sqlite3.connect("data.db") 
    # except: 
    #     raise FileNotFoundError("Can't create database")
    

    t0 = time.time()
    out = []
    for d in date_range:
        out.append(pl.DataFrame(get_reg_goals(d)))
    t1 = time.time()

    df = pl.concat(out, how = "diagonal")

    df.write_database(
        table_name = "goal_data",
        connection = f"sqlite://{path_to_db}//data.db",
        if_table_exists="replace"
    )



def update_database(path_to_db: str) -> None:
    pass


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



    # out = build_database("2023-10-04", "")
    # print(out)