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


def get_game_ids(date: str) -> pl.DataFrame:
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    response = requests.get(url)
    data = response.json()

    games = data['gameWeek'][0]['games']

    # Extract the relevant information and create a DataFrame
    out = pl.DataFrame([{
        
        "game_id": game['id'],
        "date": game['date'],
        "home_team": game['homeTeam']['abbrev'],
        "away_team": game['awayTeam']['abbrev'],
        "season": game['season']
    } for game in games])

    return out


def get_model_data(path_to_db: str, season: str, max_date: str) -> dict[str, ]:
    con = sqlite3.connect(f"{path_to_db}/data.db") 
    query = f"""
        SELECT date, CAST(id as text) game_id, away_team, 
                home_team, home_goals, away_goals
        FROM goal_data
        WHERE date < "{max_date}" AND substr(game_id, 1, 4) == "{season}"
        ORDER BY game_id
    """
    out = (
        pl.read_database(query=query, connection=con)
    )
    
    print(out)
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
        "home_teams": result["model_df"]["home_id"].to_numpy(),
        "away_teams": result["model_df"]["away_id"].to_numpy(),
        "home_goals": result["model_df"]["home_goals"].to_numpy(),
        "away_goals": result["model_df"]["away_goals"].to_numpy(),
        "home_new": result["team_id_map"].filter(pl.col("team") == home_team).select("id").to_list(),
        "away_new": result["team_id_map"].filter(pl.col("team") == away_team).select("id").to_list(),
        "N_new": 1
    }

    model_fit = model.sample(datalist, parallel_chains=4)

    return model_fit

result = get_model_data("data", "2023", "2024-02-01")

datalist = {
    "N": result["model_df"].shape[0],
    "n_teams": result["team_id_map"].shape[0],
    "home_teams": result["model_df"]["home_id"].to_numpy(),
    "away_teams": result["model_df"]["away_id"].to_numpy(),
    "home_goals": result["model_df"]["home_goals"].to_numpy(),
    "away_goals": result["model_df"]["away_goals"].to_numpy(),
    "home_new": result["team_id_map"].filter(pl.col("team") == "TOR")["id"].to_numpy(),
    "away_new": result["team_id_map"].filter(pl.col("team") == "BOS")["id"].to_numpy(),
    "N_new": 1
}

# model = cmdstanpy.CmdStanModel(stan_file = "src/model/model.stan")
# model_fit = model.sample(datalist, parallel_chains=4)

# home_new_draw = model_fit.stan_variable("home_new")
# away_new_draw = model_fit.stan_variable("away_new")

# draws_df = pl.DataFrame({
#     "home_new_draw": home_new_draw,
#     "away_new_draw": away_new_draw
# })