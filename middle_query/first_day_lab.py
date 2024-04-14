from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH
from middle_query import blood_differential, chemistry, coagulation, complete_blood_count, enzyme
from query_exceptions import ResultEmptyException
from extract_target_patients import extractTargetPatients


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH)

    dfPatients = extractTargetPatients()
    
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
    result.to_csv(OUTPUT_PATH)

    return result
