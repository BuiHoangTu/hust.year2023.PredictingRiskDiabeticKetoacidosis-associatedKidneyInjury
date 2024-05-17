from pathlib import Path
import pandas as pd
from constants import queryPostgresDf

from constants import TEMP_PATH,MIMIC_PATH
from utils.query_exceptions import ResultEmptyException
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu,getTargetPatientAdmission,getTargetPatientIcd
from utils.extract_mesurements import extractChartEventMesures
from mimic_sql import bg
from mimic_sql import ventilation
from mimic_sql import gcs
from mimic_sql import vitalsign
from mimic_sql import urine_output
from mimic_sql import chemistry
from mimic_sql import complete_blood_count
from mimic_sql import enzyme
from mimic_sql import age


def runSql():
    THIS_FILE = Path(__file__)

    OUTPUT_PATH = TEMP_PATH / (THIS_FILE.name + ".csv")

    if (OUTPUT_PATH).exists():
        return pd.read_csv(OUTPUT_PATH, parse_dates=["starttime", "endtime"])

    CHART_EVENT_IDs = [
        226732
    ]

    dfPatients = getTargetPatientIcu()
    dfAdmission= getTargetPatientAdmission()
    dfChartevent=extractChartEventMesures(CHART_EVENT_IDs,  "charted_" + THIS_FILE.name + ".csv")

    dfServices = pd.read_csv(MIMIC_PATH / "hosp" / "services.csv")
    dfDiagnosesIcd = getTargetPatientIcd()
    result = pd.DataFrame()
    queryStr = (Path(__file__).parent /  (THIS_FILE.stem + ".sql")).read_text()
    queryStr = queryStr.replace("%", "%%")
    map = {
        "icustays": dfPatients,
        "chartevents": dfChartevent,
        "admissions": dfAdmission,
        "services": dfServices,
        "diagnoses_icd": dfDiagnosesIcd,
        "bg": bg.runSql(),
        "ventilation": ventilation.extractVentilation(),
        "gcs": gcs.runSql(),
        "vitalsign": vitalsign.runSql(),
        "urine_output":urine_output.extractUrineOutput() ,
        "chemistry": chemistry.runSql(),
        "complete_blood_count": complete_blood_count.runSql(),
        "enzyme": enzyme.runSql(),
        "age": age.runSql(),
    }
    result = queryPostgresDf(queryStr, map)
    pass

    if result is None:
        raise ResultEmptyException()
    result.to_csv(OUTPUT_PATH)

    return result
