from pathlib import Path
import pandas as pd
from constants import MIMIC_PATH, queryPostgresDf

from constants import TEMP_PATH
from mimic_sql import SQL_FOLDER, SqlWrapper


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.stem + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["admittime"])

    else:
        sqlWrapper = SqlWrapper(
            queryPostgresDf,
            sqlPath=SQL_FOLDER,
            sqlFileName=THIS_FILE.stem + ".sql",
        )

        sqlWrapper["admissions"] = pd.read_csv(
            MIMIC_PATH / "hosp" / "admissions.csv", parse_dates=["admittime"]
        )
        sqlWrapper["patients"] = pd.read_csv(MIMIC_PATH / "hosp" / "patients.csv")

        return sqlWrapper.runSQL()
