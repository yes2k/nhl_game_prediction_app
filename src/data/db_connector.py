import mysql.connector
import json 

with open('./data/database_creds.json', 'r') as f:
    database_creds = json.load(f)


class DBConnector:
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host=database_creds["host"], 
            user=database_creds["user"], 
            password=database_creds["password"],
            allow_local_infile=True
        )
    
    