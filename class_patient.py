from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Collection, Dict, List, Tuple
from numpy import datetime64
from pandas import DataFrame, Timestamp, to_datetime
from sortedcontainers import SortedDict


class Patient:

    def __init__(self, subjectId: int, hadmId: int, stayId: int) -> None:
        self.subjectId = subjectId
        self.hadmId = hadmId
        self.stayId = stayId
        self.measures: Dict[str, Dict[Timestamp, float]] = SortedDict()
        pass

    def putMeasure(
        self,
        measureName: str,
        measureTime: str | datetime | datetime64 | Timestamp,
        measureValue: float,
    ):
        measureTime = to_datetime(measureTime)

        measure = self.measures.get(measureName)

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

        return df

    def toJson(self):
        jsonData = {
            "subjectId": self.subjectId,
            "hadmId": self.hadmId,
            "stayId": self.stayId,
            "measures": {},
        }
        for measureName, measureData in self.measures.items():
            jsonData["measures"][measureName] = {}
            for timestamp, value in measureData.items():
                jsonData["measures"][measureName][timestamp.isoformat()] = value
        return jsonData


def toJsonFile(patients: Collection[Patient], file: str | Path):
    jsonData = []
    for obj in patients:
        jsonData.append(obj.toJson())

    Path(file).write_text(json.dumps(jsonData, indent=4))


def fromJsonFile(file: str | Path):
    file = Path(file)

    jsonData = json.loads(file.read_text())
    patients = []
    for item in jsonData:
        patient = Patient(item["subjectId"], item["hadmId"], item["stayId"])
        for measureName, measureData in item["measures"].items():
            for timestampStr, value in measureData.items():
                timestamp = Timestamp(timestampStr)
                patient.putMeasure(measureName, timestamp, value)
        patients.append(patient)
    return patients
