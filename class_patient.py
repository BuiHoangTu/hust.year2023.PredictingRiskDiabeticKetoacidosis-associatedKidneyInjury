from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Collection, Dict, List, Tuple
import numpy as np
from numpy import datetime64
from pandas import DataFrame, Timestamp, to_datetime
from sortedcontainers import SortedDict
from constants import TEMP_PATH
from target_patients import getTargetPatientIcu
import akd_positive
from variables.charateristics_diabetes import (
    getDiabeteType,
    getMacroangiopathy,
    getMicroangiopathy,
)
from variables.demographics import getAge, getEthnicity, getGender, getHeight, getWeight
from variables.interventions import getMV, getNaHCO3
import variables.lab_test as lab_test
from variables.scoring_systems import getGcs, getOasis, getSofa, getSaps2
from variables.vital_signs import (
    getHeartRate,
    getRespiratoryRate,
    getSystolicBloodPressure,
    getDiastolicBloodPressure,
)
from variables.prognosis import getPreIcuLos
from variables.comorbidities import (
    getHistoryACI,
    getHistoryAMI,
    getCHF,
    getLiverDisease,
    getPreExistingCKD,
    getMalignantCancer,
    getHypertension,
    getUTI,
    getChronicPulmonaryDisease,
)


DEFAULT_PATIENTS_FILE = TEMP_PATH / "learning_data.json"


class PatientJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(PatientJsonEncoder, self).default(obj)


class Patient:

    def __init__(
        self,
        subjectId: int,
        hadmId: int,
        stayId: int,
        akdPositive: bool,
        measures: Dict[str, Dict[Timestamp, float] | float] | None = None,
    ) -> None:
        self.subjectId = subjectId
        self.hadmId = hadmId
        self.stayId = stayId
        self.akdPositive = akdPositive
        self.measures: Dict[str, Dict[Timestamp, float] | float] = SortedDict()

        if measures is None:
            return 
        # parse measures
        for key, value in measures.items():
            if isinstance(value, Dict):
                for mTime, mValue in value.items():
                    self.putMeasure(key, mTime, mValue)
                    pass
                pass
            else:
                self.putMeasure(key, None, value)
        pass

    def putMeasure(
        self,
        measureName: str,
        measureTime: str | datetime | datetime64 | Timestamp | None,
        measureValue: float,
    ):

        if measureTime is None:
            self.measures[measureName] = measureValue
            return 

        measureTime = to_datetime(measureTime)

        measure = self.measures.get(measureName)

        if isinstance(measure, float) or isinstance(measure, int):
            self.measures[measureName] = measureValue
            return 

        if measure is None:
            measure = self.measures[measureName] = SortedDict()
            pass

        measure[measureTime] = measureValue

    def getMeasuresBetween(
        self,
        fromTime: str | datetime | datetime64 | Timestamp,
        toTime: str | datetime | datetime64 | Timestamp,
        how: str | Callable[[DataFrame], float] = "avg",
    ):
        """Get patient's status during specified period.

        Args:
            fromTime (str | datetime | datetime64 | Timestamp): start time
            toTime (str | datetime | datetime64 | Timestamp): end time
            how : {'first', 'last', 'avg', 'max', 'min', 'std'} | Callable[[DataFrame], float], default 'avg'
                Which value to choose if multiple exist:

                    - first: Use first recored value.
                    - last: Use last recored value.
                    - avg: Use average of values.
                    - max: Use max value.
                    - min: Use min value.
                    - std: Use standard deviation of values
                    - custom function that take dataframe(time, value) and return value

        Returns:
            DataFrame: one row with full status of patient
        """

        # unify input
        fromTime = to_datetime(fromTime)
        toTime = to_datetime(toTime)
        howMapping: Dict[str, Callable[[DataFrame], float]] = {
            "first": lambda df: df.loc[df["time"].idxmin(), "value"],  # type: ignore
            "last": lambda df: df.loc[df["time"].idxmax(), "value"],
            "avg": lambda df: df["value"].mean(),
            "max": lambda df: df["value"].max(),
            "min": lambda df: df["value"].min(),
            "std": lambda df: df["value"].std(),
        }
        if how in howMapping:
            how = howMapping[how]

        if not isinstance(how, Callable): 
            raise Exception("Unk how: ", how)

        df = DataFrame(
            {
                "subject_id": self.subjectId,
                "hadm_id": self.hadmId,
                "stay_id": self.stayId,
            }
        )

        for measureName, measureTimeValue in self.measures.items():

            if isinstance(measureTimeValue, dict):
                measureTimes = list(measureTimeValue.keys())
                left = 0
                right = len(measureTimeValue) - 1

                while left <= right:
                    mid = left + (right - left) // 2

                    if measureTimes[mid] >= fromTime:
                        startId = mid
                        right = mid - 1

                    else:
                        left = mid + 1
                        pass
                    pass

                measureInRange: List[Tuple[Timestamp, float]] = []
                for i in range(startId, len(measureTimes)):
                    if measureTimes[i] > toTime:
                        break

                    measureInRange.append((measureTimes[i], measureTimeValue[measureTimes[i]]))
                    pass

                dfMeasures = DataFrame(measureInRange, columns=["time", "value"])
                measureValue = how(dfMeasures)

                df[measureName] = measureValue
                pass
            else:
                df[measureName] = measureTimeValue
                pass
            pass

        return df

    def toJson(self):
        jsonData = self.__dict__.copy()

        jsonData["measures"] = {}

        for measureName, measureData in self.measures.items():
            if isinstance(measureData, dict):
                jsonData["measures"][measureName] = {}
                for timestamp, value in measureData.items():
                    jsonData["measures"][measureName][timestamp.isoformat()] = value
                    pass
                pass
            else:
                jsonData["measures"][measureName] = measureData         
        return jsonData


class Patients:
    """Create a list of patients. Read from cache file if avaiable
    """

    def __init__(self, storagePath: Path | str | None = None) -> None:
        if storagePath is None:
            storagePath = DEFAULT_PATIENTS_FILE
        storagePath = Path(storagePath)
        
        self.storagePath = storagePath
        
        if storagePath.exists():
            self.patientList = Patients.fromJsonFile(storagePath)
        else:
            patientList: List[Patient] = []
            self.patientList = patientList
            
            dfPatient = getTargetPatientIcu()
            dfPatient = dfPatient[["subject_id", "hadm_id", "stay_id"]]


            dfAkd = akd_positive.extractKdigoStages7day()
            dfAkd["akd"] = dfAkd["aki_7day"]
            dfAkd = dfAkd[["stay_id", "akd"]]

            dfData1 = dfPatient.merge(dfAkd, "left", "stay_id")
            dfData1["akd"] = dfData1["akd"].astype(bool)

            for _, row in dfData1.iterrows():
                patient = Patient(row["subject_id"], row["hadm_id"], row["stay_id"], row["akd"])
                patientList.append(patient)
                pass

            dfData1["akd"].value_counts()


            ########### Characteristics of diabetes ###########
            df = getDiabeteType()
            df["dka_type"] = df["dka_type"].astype(int)
            self.putDataForPatients(df)


            df = getMacroangiopathy()
            self.putDataForPatients(df)


            df = getMicroangiopathy()
            self.putDataForPatients(df)


            ########### Demographics ###########
            df = getAge()
            self.putDataForPatients(df)


            df = getGender()
            self.putDataForPatients(df)


            df = getEthnicity()
            self.putDataForPatients(df)


            df = getHeight()
            self.putDataForPatients(df)


            df = getWeight()
            self.putDataForPatients(df)


            ########### Laboratory test ###########
            df = lab_test.getWbc().dropna()
            self.putDataForPatients(df)


            df = lab_test.getLymphocyte().dropna()
            self.putDataForPatients(df)


            df = lab_test.getHb().dropna()
            self.putDataForPatients(df)


            df = lab_test.getPlt().dropna()
            self.putDataForPatients(df)


            df = lab_test.getPO2().dropna()
            self.putDataForPatients(df)


            df = lab_test.getPCO2().dropna()
            self.putDataForPatients(df)


            df = lab_test.get_pH().dropna()
            self.putDataForPatients(df)


            df = lab_test.getAG().dropna()
            self.putDataForPatients(df)


            df = lab_test.getBicarbonate().dropna()
            self.putDataForPatients(df)


            df = lab_test.getBun().dropna()
            self.putDataForPatients(df)


            df = lab_test.getCalcium().dropna()
            self.putDataForPatients(df)


            df = lab_test.getScr().dropna()
            self.putDataForPatients(df)


            df = lab_test.getBg().dropna()
            self.putDataForPatients(df)


            df = lab_test.getPhosphate().dropna()
            self.putDataForPatients(df)


            df = lab_test.getAlbumin().dropna()
            self.putDataForPatients(df)


            df = lab_test.get_eGFR().dropna()
            self.putDataForPatients(df)


            df = lab_test.getHbA1C().dropna()
            self.putDataForPatients(df)


            df = lab_test.getCrp().dropna()
            self.putDataForPatients(df)


            df = lab_test.getUrineKetone().dropna()
            self.putDataForPatients(df)


            ########### Scoring systems ###########
            df = getGcs().dropna()
            self.putDataForPatients(df)


            df = getOasis().dropna()
            self.putDataForPatients(df)


            df = getSofa()
            self.putDataForPatients(df)


            df = getSaps2()
            self.putDataForPatients(df)


            ########### Vital signs ###########
            df = getHeartRate().dropna()
            self.putDataForPatients(df)


            df = getRespiratoryRate().dropna()
            self.putDataForPatients(df)


            df = getSystolicBloodPressure().dropna()
            self.putDataForPatients(df)


            df = getDiastolicBloodPressure().dropna()
            self.putDataForPatients(df)


            ########### Prognosis ###########
            df = getPreIcuLos().dropna()
            self.putDataForPatients(df)


            df = getHistoryACI()
            self.putDataForPatients(df)


            ########### Comorbidities ###########
            df = getHistoryAMI()
            self.putDataForPatients(df)


            df = getCHF()
            self.putDataForPatients(df)


            df = getLiverDisease()
            self.putDataForPatients(df)


            df = getPreExistingCKD()
            self.putDataForPatients(df)


            df = getMalignantCancer()
            self.putDataForPatients(df)


            df = getHypertension()
            self.putDataForPatients(df)


            df = getUTI()
            self.putDataForPatients(df)


            df = getChronicPulmonaryDisease()
            self.putDataForPatients(df)


            ########### Interventions ###########
            df = getMV()
            self.putDataForPatients(df)


            df = getNaHCO3()
            self.putDataForPatients(df)


            ########### Save file ###########
            Patients.toJsonFile(patientList, self.storagePath)


        pass
    
    def putDataForPatients(self, df):
        for patient in self.patientList:
            if "stay_id" in df.columns:
                dfIndividualMeasures = df[df["stay_id"] == patient.stayId]
            elif "hadm_id" in df.columns:
                dfIndividualMeasures = df[df["hadm_id"] == patient.hadmId]
            elif "subject_id" in df.columns:
                dfIndividualMeasures = df[df["subject_id"] == patient.subjectId]
            else:
                print("DataFrame does not have 'hadm_id' or 'stay_id' column.")
                return

            dfIndividualMeasures = dfIndividualMeasures.reset_index(drop=True)

            dataColumns = [
                x
                for x in dfIndividualMeasures.columns
                if x not in ["stay_id", "hadm_id", "time"]
            ]

            for _, row in dfIndividualMeasures.iterrows():
                for dataColumn in dataColumns:
                    patient.putMeasure(dataColumn, row.get("time"), row[dataColumn])

    def savePatients(self, patients: Collection[Patient]):
        self.patientList = patients
        
        Patients.toJsonFile(patients, self.storagePath)


    @staticmethod
    def toJsonFile(patients: Collection[Patient], file: str | Path):
        jsonData = []
        for obj in patients:
            jsonData.append(obj.toJson())

        Path(file).write_text(json.dumps(jsonData, indent=4, cls=PatientJsonEncoder))

    @staticmethod
    def fromJsonFile(file: str | Path):
        file = Path(file)

        jsonData: List[Dict] = json.loads(file.read_text())        
        return [Patient(**d) for d in jsonData]
