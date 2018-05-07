from fact.factdb import connect_database, RunInfo, get_ontime_by_source_and_runtype, get_ontime_by_source, Source, SourceType, read_into_dataframe, RunType
from fact.credentials import get_credentials
import os
from fact.io import to_h5py

os.environ["FACT_PASSWORD"] = "40drs4320Bias"

connect_database()

num_runs = RunInfo.select().count()
print(num_runs)
runInfo = read_into_dataframe(RunInfo.select())
source = read_into_dataframe(Source.select())
sourcetype = read_into_dataframe(SourceType.select())
runtype = read_into_dataframe(RunType.select())

to_h5py(runInfo, "RunInfo.hdf5", key="info")
to_h5py(source, "Source.hdf5", key="info")
to_h5py(runtype, "RunType.hdf5", key="info")
to_h5py(sourcetype, "SourceType.hdf5", key="info")
