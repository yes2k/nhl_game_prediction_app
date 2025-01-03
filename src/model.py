from datetime import date, datetime, timedelta
import polars as pl
import cmdstanpy
import sqlite3
import requests

def get_all_teams(date: str) -> pl.DataFrame:
    """
        date has to be formatted like YYYY-MM-DD
    """

    # TODO: figure out a better way to get all the teams for a specific season
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

    model_fit = model.sample(datalist, parallel_chains=4, show_console=True)

    return ({
        "model_fit": model_fit,
        "datalist": datalist,
        'team_id_map': result["team_id_map"]
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
    prob_home_team_ot_win = model_dict["model_fit"].stan_variable("home_ot_win_prob")

    pred_df = (
        pl.DataFrame({
            "home_new_draw": home_new_draw,
            "away_new_draw": away_new_draw,
            "prob_home_team_ot_win": prob_home_team_ot_win
        }).explode("home_new_draw", "away_new_draw", "prob_home_team_ot_win")
        .rename({"home_new_draw": "home", "away_new_draw": "away"})
    )

    combination_counts = (
        pred_df.group_by(["home", "away"]).len().sort(["home", "away"])
        .with_columns((pl.col("len") / pl.col("len").sum()) * 100)
    )

    # Getting probability of home team win
    prob_home_team_win = (
        pred_df
        .with_columns(
            pl.when(pl.col("home") > pl.col("away")).then(1)
            .when(pl.col("home") < pl.col("away")).then(0)
            .when(pl.col("home") == pl.col("away"))
            .then((pl.col("prob_home_team_ot_win") > 0.5).cast(pl.Int32)).alias("home_team_win")
        )["home_team_win"].mean()
    )

    return {'combination_counts': combination_counts, 'prob_home_team_win': prob_home_team_win * 100}



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

    pred = get_table_of_predictions(model_out)

    return {
        'table_of_pred': pred['combination_counts'],
        'prob_home_team_win': pred['prob_home_team_win'],
        'team_params': (
            get_team_latent_params(model_out)
            .join(
                model_out['team_id_map'],
                right_on="id",
                left_on="team_id",
                how="left"
            )
            .drop('team_id')
            .sort("team")
            .rename({'5%': 'lower_5p', '50%': 'median', '95%': 'upper_95p'})
        )
    }



def get_season_log_loss(season: str) -> pl.DataFrame:
    if season not in ['2023', '2024']: 
        raise ValueError(f'{season} is not avaliable')

    con = sqlite3.connect("data/data.db") 
    query = f"""
        SELECT date, CAST(id as text) game_id, away_team, 
                home_team, home_goals, away_goals, winning_team
        FROM goal_data
        WHERE substr(game_id, 1, 4) == '{season}'
        ORDER BY game_id
    """
    query_df = (
        pl.read_database(query=query, connection=con)
    )


    out = (
        query_df
        .slice(1, query_df.height - 1)
        .with_columns(
            pl.struct("date", "home_team", "away_team")
            .map_elements(lambda x: get_prediction(x["date"], x["home_team"], x["away_team"])["prob_home_team_win"], 
                            return_dtype = pl.Float64)
            .alias("prob_home_team_win")
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