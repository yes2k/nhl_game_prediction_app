from datetime import date, datetime, timedelta
import polars as pl
import cmdstanpy
import sqlite3
import requests
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
import os
import json

# import src.helper as helper
import helper as helper


@dataclass
class DataModel:
    model_df: pl.DataFrame
    team_id_map: pl.DataFrame

@dataclass
class ModelResult:
    model: cmdstanpy.CmdStanMCMC 
    model_data: DataModel

@dataclass
class PredResult:
    pred_table: pl.DataFrame
    prob_home_team_win: pl.DataFrame
    team_params: pl.DataFrame


class GamePredModel:
    
    def __init__(self, path_to_db, path_to_model):
        self.path_to_db = path_to_db
        self.path_to_model = path_to_model


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

        if out.shape[0] == 0:
            raise IndexError("No Data Found")

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


    def __fit_model(self, max_date: str, season: str, home_team: str, away_team: str) -> ModelResult:
        # Loading stan model from file
        model = cmdstanpy.CmdStanModel(stan_file = self.path_to_model)

        dat = self.__get_model_data(max_date, season)
        training_data = {
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

        # Fitting model
        model_fit = model.sample(training_data, parallel_chains=4)

        return ModelResult(model_fit, dat)


    def __fit_model_multiple_preds(self, max_date: str, season: str, home_teams: list[str], away_teams: list[str]) -> ModelResult:
        if len(home_teams) != len(away_teams):
            raise IndexError("len(home_teams) != len(away_teams)") 


        # Loading stan model from file
        model = cmdstanpy.CmdStanModel(stan_file = self.path_to_model)

        dat = self.__get_model_data(max_date, season)
        training_data = {
            "N": dat.model_df.shape[0], 
            "n_teams": dat.team_id_map.shape[0],
            "home_teams": dat.model_df["home_id"].to_list(),
            "away_teams": dat.model_df["away_id"].to_list(),
            "home_goals": dat.model_df["home_goals"].to_list(),
            "away_goals": dat.model_df["away_goals"].to_list(),
            "home_new": (pl.DataFrame({"team": home_teams})
                         .join(dat.team_id_map, on = "team", how = "left")["id"].to_list()),
            "away_new": (pl.DataFrame({"team": away_teams})
                         .join(dat.team_id_map, on = "team", how = "left")["id"].to_list()),
            "N_new": len(home_teams)
        }

        # Fitting model
        model_fit = model.sample(training_data, parallel_chains=4)

        return ModelResult(model_fit, dat)


    def __get_params(self, model: ModelResult, team_id_map: pl.DataFrame) -> pl.DataFrame:
        att_var = [f"att[{i}]" for i in range(1, team_id_map.shape[0] + 1)]
        def_var = [f"def[{i}]" for i in range(1, team_id_map.shape[0] + 1)]

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
            .join(team_id_map, left_on="team_id", right_on="id")
        )

        return team_latent_params


    def get_team_params(self) -> pl.DataFrame:
        con = sqlite3.connect(self.path_to_db)
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_params';")
        table_exists = cursor.fetchone()

        if table_exists:
            query = "SELECT * FROM team_params"
            return pl.read_database(query=query, connection=con)
        else:
            today_date = datetime.now().strftime("%Y-%m-%d")
            model = self.__fit_model(today_date, helper.get_nhl_season(today_date), "TOR", "BOS")
            latent_team_params = self.__get_params(model, model.model_data.team_id_map)
            return latent_team_params
    

    def get_prediction(self, max_date: str, season: str, home_team: str, away_team: str) -> PredResult:
        
        con = sqlite3.connect(self.path_to_db)
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pred_goal_data';")
        table_exists = cursor.fetchone()

        if table_exists:
            query = f"""
                SELECT date_of_game as date, CAST(game_id as text) game_id, away_team, 
                    home, away, prob_home_team_win, len
                FROM pred_goal_data
                WHERE date == "{max_date}" AND home_team == "{home_team}" AND away_team == "{away_team}"
                ORDER BY game_id
            """
            out = pl.read_database(query=query, connection=con)

            if out.shape[0] > 0:
                return PredResult(out.drop("prob_home_team_win"), out["prob_home_team_win"][0], pl.DataFrame())


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
        latent_team_params = self.__get_params(model, model.model_data.team_id_map)

        return PredResult(
            combination_counts,
            prob_home_team_win,
            latent_team_params
        )


    def get_log_loss(self):
        pass


    def get_accuracy(self):
        pass


    def get_season_prediction(self) -> dict[str, int]:
        
        if os.path.exists("data/seasons_proj.json"):
            with open("data/seasons_proj.json", "r") as f:
                return json.load(f)


        today_date = datetime.now().strftime("%Y-%m-%d")
        games_to_sim = helper.get_reg_scheduled_games(today_date, "2025-04-17")
        current_standings = helper.get_current_standings()
        current_points = dict(zip(current_standings["team"], current_standings["points"]))


        res = self.__fit_model_multiple_preds(today_date, "2024", games_to_sim["home_team"].to_list(), games_to_sim["away_team"].to_list())
        pred_home_goals = res.model.stan_variable("pred_home_goals")
        pred_away_goals = res.model.stan_variable("pred_away_goals")
        home_ot_win_prob = res.model.stan_variable("home_ot_win_prob")


        # Defs
        # W -> 2
        # L -> 3
        # OT -> 4
        # OTW -> 5
        # OTL -> 6

        home_sim_res = np.where(pred_home_goals > pred_away_goals, 2, np.where(pred_home_goals < pred_away_goals, 3, 4))
        home_sim_res[np.where(home_sim_res == 4)] = np.random.binomial(1, home_ot_win_prob[np.where(home_sim_res == 4)])
        home_sim_res = np.where(home_sim_res == 1, 5, home_sim_res)
        home_sim_res = np.where(home_sim_res == 0, 6, home_sim_res)

        away_sim_res = np.where(home_sim_res == 2, 3, np.where(home_sim_res == 3, 2, np.where(home_sim_res == 5, 6, 5)))
        
        away_sim_res = np.where(away_sim_res == 2, 2, np.where(away_sim_res == 3, 0, np.where(away_sim_res == 5, 2, 1)))
        home_sim_res = np.where(home_sim_res == 2, 2, np.where(home_sim_res == 3, 0, np.where(home_sim_res == 5, 2, 1)))

        team_point_proj = {}
        for team in set(games_to_sim["home_team"].unique().to_list() + games_to_sim["away_team"].unique().to_list()):
            team_home_games_idx = (
                (games_to_sim["home_team"] == team)
            ).arg_true().to_list()

            team_away_games_idx = (
                (games_to_sim["away_team"] == team)
            ).arg_true().to_list()

            team_point_proj[team] = (home_sim_res[:,team_home_games_idx].sum(axis = 1) + away_sim_res[:,team_away_games_idx].sum(axis = 1)) + current_points[team]
            team_point_proj[team] = team_point_proj[team].tolist()
        return team_point_proj
    

    def get_prediction_heatmap_html(self, max_date: str, season: str, home_team: str, away_team: str) -> str:
        pred = self.get_prediction(max_date, season, home_team, away_team)

        fig = go.Figure(data=go.Heatmap(
            z=pred.pred_table["len"].to_list(),
            x=pred.pred_table["home"].to_list(),
            y=pred.pred_table["away"].to_list(),
            hoverongaps=False,
            hovertemplate='Home Goals: %{x}<br>Away Goals: %{y}<br>Probability: %{z}<extra></extra>'
            )
        )

        fig = fig.add_annotation(     
            x=max(pred.pred_table["home"].to_list()) - 1,
            y=max(pred.pred_table["away"].to_list()) - 1,
            text=f"{home_team} Win Probability: {round(pred.prob_home_team_win * 100, 2)}%",
            showarrow=False,
            font=dict(
                size=16,
                color="#ffffff"
            ),
            align="center",
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
        )


        fig.update_layout(
            xaxis_title=f"{home_team} Goals",
            yaxis_title=f"{away_team} Goals"
        )

        return fig.to_html(full_html=False)
    

    def get_season_projection_box_plot(self):
        season_proj = self.get_season_prediction()
        fig = go.Figure()

        sorted_teams = sorted(season_proj.items(), key=lambda x: np.median(x[1]), reverse=True)
        
        for team, points in sorted_teams:
            fig.add_trace(go.Box(
            y=points,
            name=team,
            showlegend=False
            ))

        fig.update_layout(
            title="Season Points Projection",
            yaxis_title="Points",
            xaxis_title="Teams"
        )

        return fig.to_html(full_html=False)