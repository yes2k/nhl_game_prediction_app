from datetime import date, datetime, timedelta
import polars as pl
import cmdstanpy
import sqlite3
import requests

def get_all_teams(date: str) -> pl.DataFrame:
    """
        date has to be formatted like YYYY-MM-DD
    """
    url = f"https://api-web.nhle.com/v1/standings/{date}"

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


def get_model_data(path_to_db: str, season: str, max_date: str) -> dict[str, ]:
    con = sqlite3.connect(path_to_db) 
    query = f"""
        SELECT date, CAST(id as text) game_id, away_team, 
                home_team, home_goals, away_goals
        FROM goal_data
        WHERE date <= "{max_date}" AND substr(game_id, 1, 4) == "{season}"
        ORDER BY game_id
    """
    out = (
        pl.read_database(query=query, connection=con)
    )
    
     # Create team_id_map
    team_id_map = get_all_teams(max_date)

    print(out)

    # Join out with team_id_map for home and away teams
    out = (
        out
        .join(
            team_id_map.rename({"team": "home_team", "id": "home_id"}),
            on="home_team",
            how="left"
        )
        .join(
            team_id_map.rename({"team": "away_team", "id": "away_id"}),
            on="away_team",
            how="left"
        )
    )

    # Create the final list
    result = {
        "model_df": out,
        "team_id_map": team_id_map
    }

    return result


def fit_model(
        path_to_model: str, 
        path_to_db: str, 
        max_date: str, 
        season: str, 
        home_team: str, 
        away_team: str
    ):
    model = cmdstanpy.CmdStanModel(stan_file = path_to_model)
    result = get_model_data(path_to_db, season, max_date)
    
    datalist = {
        "N": result["model_df"].shape[0],
        "n_teams": result["team_id_map"].shape[0],
        "home_teams": result["model_df"]["home_id"].to_list(),
        "away_teams": result["model_df"]["away_id"].to_list(),
        "home_goals": result["model_df"]["home_goals"].to_list(),
        "away_goals": result["model_df"]["away_goals"].to_list(),
        "home_new": result["team_id_map"].filter(pl.col("team") == home_team)["id"].to_list(),
        "away_new": result["team_id_map"].filter(pl.col("team") == away_team)["id"].to_list(),
        "N_new": 1
    }

    model_fit = model.sample(datalist, parallel_chains=4)

    return ({
        "model_fit": model_fit,
        "datalist": datalist
    })


def get_team_latent_params(model_dict) -> pl.DataFrame:
    att_var = [f"att[{i}]" for i in range(1, model_dict["datalist"]["n_teams"]+1)]
    def_var = [f"def[{i}]" for i in range(1, model_dict["datalist"]["n_teams"]+1)]

    team_latent_params = (
        pl.concat([
            (
                pl.DataFrame(model_dict["model_fit"].summary().loc[att_var, ["5%", "50%", "95%"]])
                .with_columns(pl.Series("att_var", att_var).alias("var"))
            ),
            (
                pl.DataFrame(model_dict["model_fit"].summary().loc[def_var, ["5%", "50%", "95%"]])
                .with_columns(pl.Series("def_var", def_var).alias("var"))
            )
        ])
        .with_columns(pl.col("var").str.extract(r"(\d+)").cast(pl.Int32).alias("team_id"))
        .with_columns(pl.col("var").str.extract(r"(att|def)").alias("type"))
        .drop("var")
    )

    return team_latent_params


def get_table_of_predictions(model_dict) -> pl.DataFrame:
    home_new_draw = model_dict["model_fit"].stan_variable("pred_home_goals")
    away_new_draw = model_dict["model_fit"].stan_variable("pred_away_goals")

    pred_df = (
        pl.DataFrame({
            "home_new_draw": home_new_draw,
            "away_new_draw": away_new_draw
        }).explode("home_new_draw", "away_new_draw")
        .rename({"home_new_draw": "home", "away_new_draw": "away"})
    )

    prob_home_team_ot_win = model_dict["model_fit"].stan_variable("home_ot_win_prob")
    combination_counts = pred_df.group_by(["home", "away"]).len().sort("len", descending=True)

    return {
        "combination_counts": combination_counts.to_dicts()
    }



def get_prediction(date_of_pred: str, home_team: str, away_team: str) -> pl.DataFrame:
    res = get_game_ids(date_of_pred)
    if len(res) == 0:
        return {
            "error": "No games found"
        }
    else:
        season = str(res['res'][0]["season"])[:4]

    max_date = date(
        int(date_of_pred[0:4]), 
        int(date_of_pred[5:7]), 
        int(date_of_pred[8:10])
    ) - timedelta(days=1)


    model_out = fit_model(
        "src/model/model.stan",
        "data/data.db",
        home_team=home_team,
        away_team=away_team,
        season=season,
        max_date=max_date.strftime("%Y-%m-%d")
    )

    return {'res': get_table_of_predictions(model_out)}


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


# print(get_game_ids("2024-10-12"))
# print(get_prediction(
#     "2024-03-10",
#     "TOR",
#     "MTL"
# ))