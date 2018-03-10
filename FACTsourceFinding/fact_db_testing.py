from fact.factdb import connect_database, RunInfo, get_ontime_by_source_and_runtype, get_ontime_by_source
from fact.credentials import get_credentials
import os

os.environ["FACT_PASSWORD"] = "password"

connect_database()

num_runs = RunInfo.select().count()
print(num_runs)
print(get_ontime_by_source_and_runtype())
print(get_ontime_by_source())
