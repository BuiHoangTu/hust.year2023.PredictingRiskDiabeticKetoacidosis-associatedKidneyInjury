from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from mimic_sql import blood_differential, chemistry, coagulation, complete_blood_count, enzyme
from utils.query_exceptions import ResultEmptyException
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    dfPatients = getTargetPatientIcu()

    dfBloodCount = complete_blood_count.runSql()
    dfBloodCount["charttime"] = pd.to_datetime(dfBloodCount["charttime"])

    dfChem = chemistry.runSql()
    dfChem["charttime"] = pd.to_datetime(dfChem["charttime"])

    dfBloodDiff = blood_differential.runSql()
    dfBloodDiff["charttime"] = pd.to_datetime(dfBloodDiff["charttime"])

    dfCoa = coagulation.runSql()
    dfCoa["charttime"] = pd.to_datetime(dfCoa["charttime"])

    dfEnzyme = enzyme.runSql()
    dfEnzyme["charttime"] = pd.to_datetime(dfEnzyme["charttime"])   

    map = {
        "icustays": dfPatients,
        "complete_blood_count": dfBloodCount,
        "chemistry": dfChem,
        "blood_differential": dfBloodDiff,
        "coagulation": dfCoa,
        "enzyme": dfEnzyme,
    }
    result = queryPostgresDf(
        (Path(__file__).parent / (THIS_FILE.stem + ".sql")).read_text(), map
    )

    if result is None:
        raise ResultEmptyException()

    df = result
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.groupby("stay_id").agg(lambda x: x.mean()).reset_index()

    df.to_csv(OUTPUT_PATH)
    return df
