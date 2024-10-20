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
    
     # Create team_id_map
    team_id_map = get_all_teams("2024-10-12")

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

    model_dat = {
        "n_teams": result["team_id_map"].shape[0],
        "home_teams": result["model_df"]["home_id"].to_numpy(),
        "away_teams": result["model_df"]["away_id"].to_numpy(),
        "home_goals": result["model_df"]["home_goals"].to_numpy(),
        "away_goals": result["model_df"]["away_goals"].to_numpy(),
    }

    return model_dat


def fit_model(
        path_to_model: str, 
        path_to_db: str, 
        max_date: str, 
        season: str, 
        home_team: str, 
        away_team: str
    ):
    pass
    

# print(get_model_data("data", "2024", "2024-10-12"))