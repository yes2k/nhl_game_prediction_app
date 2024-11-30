from datetime import date, datetime, timedelta
import polars as pl
import cmdstanpy
import sqlite3
import requests
import src.model as model

def get_model_data(path_to_db: str, season: str, max_date: str) -> dict[str, ]:
    con = sqlite3.connect(path_to_db) 
    query = f"""
        SELECT date, CAST(id as text) game_id, away_team, 
                home_team, winning_team
        FROM goal_data
        WHERE date <= "{max_date}" AND substr(game_id, 1, 4) == "{season}"
        ORDER BY game_id
    """
    out = (
        pl.read_database(query=query, connection=con)
        .with_columns(
            pl.when(pl.col("winning_team") == pl.col("away_team"))
            .then(pl.col("home_team"))
            .otherwise(pl.col("away_team"))
            .alias("losing_team")
        ).select(["game_id", "date", "winning_team", "losing_team"])
    )
    
    # Create team_id_map
    team_id_map = model.get_all_teams(max_date)

    # Join out with team_id_map for home and away teams
    out = (
        out
        .join(
            team_id_map.rename({"team": "losing_team", "id": "losing_id"}),
            on="losing_team",
            how="left"
        )
        .join(
            team_id_map.rename({'team': 'winning_team', 'id': 'winning_id'}),
            on='winning_team',
            how='left'
        )
    )

    # Create the final list
    result = {
        "model_df": out,
        "team_id_map": team_id_map.with_columns(pl.col("id").cast(pl.Int32))
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
        "n_games": result["model_df"].shape[0],
        "n_teams": result["team_id_map"].shape[0],
        "winner_id": result["model_df"]["winning_id"].to_list(),
        "loser_id": result["model_df"]["losing_id"].to_list(),
        "pred_t1_id": result["team_id_map"].filter(pl.col("team") == home_team)["id"].to_list(),
        "pred_t2_id": result["team_id_map"].filter(pl.col("team") == away_team)["id"].to_list(),
        "n_pred": 1
    }
    
    model_fit = model.sample(datalist, parallel_chains=4)

    return({
        "model_fit": model_fit,
        "datalist": datalist,
        "team_id_map": result["team_id_map"]
    })


def get_prediction(date_of_pred: str, home_team: str, away_team: str) -> pl.DataFrame:
    res = model.get_game_ids(date_of_pred)
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
        "src/model/bt_model.stan",
        "data/data.db",
        season=season,
        max_date=max_date,
        home_team=home_team,
        away_team=away_team
    )

    prob_home_team_win = model_out["model_fit"].stan_variable("t1_pred")
    return prob_home_team_win






