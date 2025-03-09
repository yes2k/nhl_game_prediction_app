import datetime

import model as model
# import src.helper as helper
import database_helper as dbh
import time


t0 = time.time()
dbh.build_database("2024-10-04", "data/")
t1 = time.time()
print(t1-t0)

# dbh.update_database("test/")


Mod = model.GamePredModel("data/data.db", "src/model/model.stan")
# print(Mod.get_team_params())
# Mod.get_season_prediction()

print(Mod.get_prediction("2025-03-08", "2024", "OTT", "NYR").pred_table)

# today_date = datetime.datetime.now().strftime("%Y-%m-%d")
# print(helper.get_scheduled_games(today_date, "2025-04-17"))