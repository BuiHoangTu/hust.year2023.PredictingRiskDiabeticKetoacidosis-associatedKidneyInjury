from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Collection, Dict, Iterable, List, Literal, Tuple
import numpy as np
from numpy import datetime64
import pandas as pd
from pandas import DataFrame, Timedelta, Timestamp, to_datetime
from sklearn.model_selection import StratifiedKFold
from sortedcontainers import SortedDict
from constants import TEMP_PATH
from mimic_sql import chemistry, complete_blood_count
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu
import akd_positive
from utils.reduce_mesurements import reduceByHadmId
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
        if isinstance(obj, Timestamp):
            return obj.isoformat()
        if isinstance(obj, Timedelta):
            return obj.total_seconds()
        return super(PatientJsonEncoder, self).default(obj)


class Patient:

    def __init__(
        self,
        subject_id: int,
        hadm_id: int,
        stay_id: int,
        intime: str | datetime | datetime64 | Timestamp,
        akdPositive: bool,
        measures: Dict[str, Dict[Timestamp, float] | float] | None = None,
        akdTime: Timedelta | float = pd.Timedelta(days=10),
    ) -> None:
        if isinstance(akdTime, float) or isinstance(akdTime, int):
            akdTime = pd.Timedelta(seconds=akdTime)
        
        self.subject_id = subject_id
        self.hadm_id = hadm_id
        self.stay_id = stay_id
        self.intime = to_datetime(intime)
        self.akdPositive = akdPositive
        self.akdTime = akdTime
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

    def removeMeasures(self, measureNames: Collection[str]):
        for measureName in measureNames:
            if measureName in self.measures:
                self.measures.pop(measureName, None)
        pass

    def getMeasuresBetween(
        self,
        fromTime: pd.Timedelta,
        toTime: pd.Timedelta,
        how: str | Callable[[DataFrame], float] = "avg",
        measureTypes: Literal["all", "static", "time"] = "all",
    ):
        """Get patient's status during specified period.

        Args:
            fromTime (pd.Timedelta): start time compare to intime (icu admission)
            toTime (pd.Timedelta): end time compare to intime (icu admission)
            how : {'first', 'last', 'avg', 'max', 'min', 'std'} | Callable[[DataFrame], float], default 'avg'
                Which value to choose if multiple exist:

                    - first: Use first recored value.
                    - last: Use last recored value.
                    - avg: Use average of values.
                    - max: Use max value.
                    - min: Use min value.
                    - std: Use standard deviation of values
                    - custom function that take dataframe(time, value) and return value
            measureTypes : {'all', 'static', 'time'}, default 'all', get all measures, only static measures, only time series measures

        Returns:
            DataFrame: one row with full status of patient
        """

        # unify input
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
                "subject_id": [self.subject_id],
                "hadm_id": [self.hadm_id],
                "stay_id": [self.stay_id],
                "akd": [self.akdPositive],
            }
        )

        for measureName, measureTimeValue in self.measures.items():

            if isinstance(measureTimeValue, dict):
                if measureTypes not in ["all", "time"]:
                    continue
                
                measureTimes = list(measureTimeValue.keys())
                left = 0
                right = len(measureTimeValue) - 1

                while left <= right:
                    mid = left + (right - left) // 2

                    if measureTimes[mid] >= self.intime + fromTime:
                        startId = mid
                        right = mid - 1

                    else:
                        left = mid + 1
                        pass
                    pass

                measureInRange: List[Tuple[Timestamp, float]] = []

                try:
                    for i in range(startId, len(measureTimes)):
                        if measureTimes[i] > self.intime + toTime:
                            break

                        measureInRange.append(
                            (measureTimes[i], measureTimeValue[measureTimes[i]])
                        )
                        pass
                except UnboundLocalError:
                    pass

                dfMeasures = DataFrame(measureInRange, columns=["time", "value"])
                measureValue = how(dfMeasures)

                df[measureName] = measureValue
                pass
            else:
                if measureTypes not in ["all", "static"]:
                    continue
                
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

    def __hash__(self) -> int:
        return hash(self.stay_id)


class Patients:
    """Create a list of patients. Read from cache file if avaiable"""

    def __init__(
        self,
        patients: List[Patient],
    ) -> None:
        if patients is not None:
            self.patientList = patients
        pass

    def __getitem__(self, id) -> Patient:
        return self.patientList[id]

    def __add__(self, other):
        if isinstance(other, Patient):
            new = Patients(patients=self.patientList)
            new.patientList.append(other)
            return new
        elif isinstance(other, Iterable) and all(
            isinstance(item, Patient) for item in other
        ):
            new = Patients(patients=self.patientList)
            new.patientList.extend(other)
            return new
        elif isinstance(other, Patients):
            return Patients(patients=self.patientList + other.patientList)
        else:
            raise TypeError(
                "Unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self), type(other)
                )
            )

    def __len__(self):
        return len(self.patientList)

    def getMeasures(self):
        featureSet: Counter[str] = Counter()
        for p in self.patientList:
            featureSet.update(p.measures.keys())
        return featureSet

    def removeMeasures(self, measureNames: Collection[str]):
        for p in self.patientList:
            p.removeMeasures(measureNames)
        pass

    def fillMissingMeasureValue(
        self, measureNames: str | list[str], measureValue: float
    ):
        if isinstance(measureNames, str):
            measureNames = [measureNames]

        for measureName in measureNames:        
            for p in self.patientList:
                if measureName not in p.measures:
                    p.putMeasure(measureName, None, measureValue)

        pass

    def removePatientByMissingFeatures(self, minimumFeatureCount: int | float = 0.8):
        if isinstance(minimumFeatureCount, float):
            minimumFeatureCount = minimumFeatureCount * len(self.getMeasures())

        for p in self.patientList:
            if len(p.measures) < minimumFeatureCount:
                self.patientList.remove(p)
        pass

    def _putDataForPatients(self, df):
        for patient in self.patientList:
            if "stay_id" in df.columns:
                dfIndividualMeasures = df[df["stay_id"] == patient.stay_id]
            elif "hadm_id" in df.columns:
                dfIndividualMeasures = df[df["hadm_id"] == patient.hadm_id]
            elif "subject_id" in df.columns:
                dfIndividualMeasures = df[df["subject_id"] == patient.subject_id]
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

    def getMeasuresBetween(
        self,
        fromTime: pd.Timedelta = pd.Timedelta(hours=-6),
        toTime: pd.Timedelta = pd.Timedelta(hours=24),
        how: str | Callable[[DataFrame], float] = "avg",
        measureTypes: Literal["all", "static", "time"] = "all",
    ):
        """Get patient's status during specified period.

        Args:
            fromTime (Timedelta): start time compare to intime (icu admission)
            toTime (Timedelta): end time compare to intime (icu admission)
            how : {'first', 'last', 'avg', 'max', 'min', 'std'} | Callable[[DataFrame], float], default 'avg'
                Which value to choose if multiple exist:

                    - first: Use first recored value.
                    - last: Use last recored value.
                    - avg: Use average of values.
                    - max: Use max value.
                    - min: Use min value.
                    - std: Use standard deviation of values
                    - custom function that take dataframe(time, value) and return value
            measureTypes : {'all', 'static', 'time'}, default 'all', get all measures, only static measures, only time series measures

        Returns:
            DataFrame: one row with full status of patient
        """

        xLs = [x.getMeasuresBetween(fromTime, toTime, how, measureTypes) for x in self.patientList]

        return pd.concat(xLs)

    def split(self, n, random_state=None):
        cachedSplitFile = (
            TEMP_PATH
            / "split" /
            f"{len(self)}-{hash(self)}-{n}-{random_state}.json"
        )
        if cachedSplitFile.exists():
            splitIndexes = json.loads(cachedSplitFile.read_text())
        else:
            indexes = [i for i in range(len(self.patientList))]
            akdLabel = [i.akdPositive for i in self.patientList]

            skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=random_state)

            splitIndexes = []
            for _, splitIndex in skf.split(indexes, akdLabel):  # type: ignore
                splitIndexes.append(splitIndex)

            cachedSplitFile.parent.mkdir(parents=True, exist_ok=True)
            json.dump(splitIndexes, cachedSplitFile.open("w+"), cls=PatientJsonEncoder)

        res: List[List[Patient]] = []
        for splitIndex in splitIndexes:
            res.append([self.patientList[i] for i in splitIndex])
        return [Patients(patients=pList) for pList in res]

    def __hash__(self) -> int:
        return hash(tuple(self.patientList))

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
        return Patients([Patient(**d) for d in jsonData])

    @staticmethod
    def loadPatients(reload: bool = False):
        if reload or not DEFAULT_PATIENTS_FILE.exists():
            patientList: List[Patient] = []

            dfPatient = getTargetPatientIcu()
            dfPatient = dfPatient[["subject_id", "hadm_id", "stay_id", "intime"]]

            dfAkd = akd_positive.extractKdigoStages7day()
            dfAkd["akd"] = dfAkd["aki_7day"]
            # dfAkd = dfAkd[["stay_id", "akd"]]

            dfData1 = dfPatient.merge(dfAkd, "left", "stay_id")
            dfData1["akd"] = dfData1["akd"].astype(bool)

            for _, row in dfData1.iterrows():

                if row["aki_stage_7day"] != 0 and row["aki_stage_creat"] == row["aki_stage_7day"]:
                    akdCreatTime = pd.Timestamp(row["charttime_creat"])
                else:
                    akdCreatTime = None

                if row["aki_stage_7day"] != 0 and row["aki_stage_uo"] == row["aki_stage_7day"]:
                    akdUrineTime = pd.Timestamp(row["charttime_uo"])
                else:
                    akdUrineTime = None

                intime = pd.Timestamp(row["intime"])

                if akdCreatTime is not None and akdUrineTime is not None:
                    akdTime = min(akdCreatTime, akdUrineTime)
                    akdTime = akdTime - intime
                elif akdCreatTime is not None:
                    akdTime = akdCreatTime
                    akdTime = akdTime - intime
                elif akdUrineTime is not None:
                    akdTime = akdUrineTime
                    akdTime = akdTime - intime
                else:
                    akdTime = Timedelta(days=10)

                patient = Patient(
                    row["subject_id"],
                    row["hadm_id"],
                    row["stay_id"],
                    intime,
                    row["akd"],
                    akdTime=akdTime,
                )
                patientList.append(patient)
                pass

            dfData1["akd"].value_counts()

            patients = Patients(patients=patientList)

            ########### Characteristics of diabetes ###########
            df = getDiabeteType()
            df["dka_type"] = df["dka_type"].astype(int)
            patients._putDataForPatients(df)

            df = getMacroangiopathy()
            patients._putDataForPatients(df)

            df = getMicroangiopathy()
            patients._putDataForPatients(df)

            ########### Demographics ###########
            df = getAge()
            patients._putDataForPatients(df)

            df = getGender()
            patients._putDataForPatients(df)

            df = getEthnicity()
            patients._putDataForPatients(df)

            df = getHeight()
            patients._putDataForPatients(df)

            df = getWeight()
            patients._putDataForPatients(df)

            ########### Laboratory test ###########
            df = lab_test.getWbc().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getLymphocyte().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getHb().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getPlt().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getPO2().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getPCO2().dropna()
            patients._putDataForPatients(df)

            df = lab_test.get_pH().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getAG().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getBicarbonate().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getBun().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getCalcium().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getScr().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getBg().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getPhosphate().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getAlbumin().dropna()
            patients._putDataForPatients(df)

            df = lab_test.get_eGFR().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getHbA1C().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getCrp().dropna()
            patients._putDataForPatients(df)

            df = lab_test.getUrineKetone().dropna()
            patients._putDataForPatients(df)

            ## extra lab variables
            ### blood count
            dfBc = reduceByHadmId(complete_blood_count.runSql())
            dfBc = dfBc[
                [
                    "stay_id",
                    "hematocrit",
                    "mch",
                    "mchc",
                    "mcv",
                    "rbc",
                    "rdw"
                ]
            ].dropna()
            patients._putDataForPatients(dfBc)

            ## blood diff (missing too much )

            ## chem
            dfChem = reduceByHadmId(chemistry.runSql())
            dfChem = dfChem[
                [
                    "stay_id",
                    "chloride",
                    "sodium",
                    "potassium",
                ]
            ].dropna()
            patients._putDataForPatients(dfChem)

            ########### Scoring systems ###########
            df = getGcs().dropna()
            patients._putDataForPatients(df)

            df = getOasis().dropna()
            patients._putDataForPatients(df)

            df = getSofa()
            patients._putDataForPatients(df)

            df = getSaps2()
            patients._putDataForPatients(df)

            ########### Vital signs ###########
            df = getHeartRate().dropna()
            patients._putDataForPatients(df)

            df = getRespiratoryRate().dropna()
            patients._putDataForPatients(df)

            df = getSystolicBloodPressure().dropna()
            patients._putDataForPatients(df)

            df = getDiastolicBloodPressure().dropna()
            patients._putDataForPatients(df)

            ########### Prognosis ###########
            df = getPreIcuLos().dropna()
            patients._putDataForPatients(df)

            df = getHistoryACI()
            patients._putDataForPatients(df)

            ########### Comorbidities ###########
            df = getHistoryAMI()
            patients._putDataForPatients(df)

            df = getCHF()
            patients._putDataForPatients(df)

            df = getLiverDisease()
            patients._putDataForPatients(df)

            df = getPreExistingCKD()
            patients._putDataForPatients(df)

            df = getMalignantCancer()
            patients._putDataForPatients(df)

            df = getHypertension()
            patients._putDataForPatients(df)

            df = getUTI()
            patients._putDataForPatients(df)

            df = getChronicPulmonaryDisease()
            patients._putDataForPatients(df)

            ########### Interventions ###########
            df = getMV()
            patients._putDataForPatients(df)

            df = getNaHCO3()
            patients._putDataForPatients(df)

            ########### Save file ###########
            Patients.toJsonFile(patientList, DEFAULT_PATIENTS_FILE)

            return patients

        else:
            return Patients.fromJsonFile(DEFAULT_PATIENTS_FILE)
