from datetime import date, datetime, timedelta
import polars as pl
import cmdstanpy
import sqlite3
import requests
from dataclasses import dataclass

import helper as helper

@dataclass
class DataModel:
    model_df: pl.DataFrame
    team_id_map: pl.DataFrame

@dataclass
class ModelResult:
    model: cmdstanpy.CmdStanMCMC 
    training_data: dict[str, ]

@dataclass
class PredResult:
    pred_table: pl.DataFrame
    prob_home_team_win: pl.DataFrame
    team_params: pl.DataFrame


class GamePredModel:
    
    def __init__(self, path_to_db, path_to_model):
        self.path_to_db = path_to_db
        self.path_to_model = path_to_model


    def __get_start_and_end_of_season(self) -> tuple[str, str]:
        url = "https://api.nhle.com/stats/rest/en/season"
        
        try:
            data = requests.get(url).json()
        except Exception as e:
            print(e)
            return None

        for seasons in data:
            if seasons[id] == self.season:
                start_date = seasons[id]['regularSeasonStartDate']
                end_date = seasons[id]['regularSeasonEndDate']
            return start_date, end_date
        else:
            print("No season data found")
            return None


    def __get_model_data(self, max_date: str, season: str) -> DataModel:
        con = sqlite3.connect(self.path_to_db) 
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

        team_id_map = helper.get_all_teams()

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

        return DataModel(
            out, 
            team_id_map.with_columns(pl.col("id").cast(pl.Int32))   
        )


    def __get_model_list(self, max_date: str, season: str, home_team: str, away_team: str) -> dict[str, ]:
        dat = self.__get_model_data(max_date, season)
        return {
            "N": dat.model_df.shape[0],
            "n_teams": dat.team_id_map.shape[0],
            "home_teams": dat.model_df["home_id"].to_list(),
            "away_teams": dat.model_df["away_id"].to_list(),
            "home_goals": dat.model_df["home_goals"].to_list(),
            "away_goals": dat.model_df["away_goals"].to_list(),
            "home_new": dat.team_id_map.filter(pl.col("team") == home_team)["id"].to_list(),
            "away_new": dat.team_id_map.filter(pl.col("team") == away_team)["id"].to_list(),
            "N_new": 1
        }


    def __fit_model(self, max_date: str, season: str, home_team: str, away_team: str) -> ModelResult:
        # Loading stan model from file
        model = cmdstanpy.CmdStanModel(stan_file = self.path_to_model)

        # Getting data to be used from the model
        training_data = self.__get_model_list(max_date, season, home_team, away_team)

        # Fitting model
        model_fit = model.sample(training_data, parallel_chains=4)

        return ModelResult(model_fit, training_data)


    def __get_params(self, model: ModelResult, nteams: int) -> pl.DataFrame:
        att_var = [f"att[{i}]" for i in range(1, nteams+1)]
        def_var = [f"def[{i}]" for i in range(1, nteams+1)]

        team_latent_params = (
            pl.concat([
                (
                    pl.DataFrame(model.model.summary().loc[att_var, ["5%", "50%", "95%"]])
                    .with_columns(pl.Series("att_var", att_var).alias("var"))
                ),
                (
                    pl.DataFrame(model.model.summary().loc[def_var, ["5%", "50%", "95%"]])
                    .with_columns(pl.Series("def_var", def_var).alias("var"))
                )
            ])
            .with_columns(pl.col("var").str.extract(r"(\d+)").cast(pl.Int32).alias("team_id"))
            .with_columns(pl.col("var").str.extract(r"(att|def)").alias("type"))
            .drop("var")
        )

        return team_latent_params


    def get_prediction(self, max_date: str, season: str, home_team: str, away_team: str) -> PredResult:

        model = self.__fit_model(max_date, season, home_team, away_team)

        # extracting predicted home and away goals from the model
        home_new_draw = model.model.stan_variable("pred_home_goals")
        away_new_draw = model.model.stan_variable("pred_away_goals")
        prob_home_team_ot_win = model.model.stan_variable("home_ot_win_prob")

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

        # Get team latent params
        latent_team_params = self.__get_params(model, model.training_data["n_teams"])

        return PredResult(
            combination_counts,
            prob_home_team_win,
            latent_team_params
        )



    def get_log_loss(self):
        pass

    def get_accuracy(self):
        pass

    def get_season_prediction(self):
        pass