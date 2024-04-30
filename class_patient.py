from datetime import datetime
import json
from pathlib import Path
from typing import Collection, Dict
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
    ):
        fromTime = to_datetime(fromTime)
        toTime = to_datetime(toTime)

        df = DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "measure_name",
                "measure_time",
                "measure_value",
            ]
        )

        for measureName, measure in self.measures.items():

            measureTimes = list(measure.keys())
            left = 0
            right = len(measure) - 1

            while left <= right:
                mid = left + (right - left) // 2

                if measureTimes[mid] >= fromTime:
                    startId = mid
                    right = mid - 1

                else:
                    left = mid + 1
                    pass
                pass

            for i in range(startId, len(measureTimes)):
                if measureTimes[i] > toTime:
                    break

                df.loc[len(df)] = [
                    self.subjectId,
                    self.hadmId,
                    self.stayId,
                    measureName,
                    measureTimes[i],
                    measure[measureTimes[i]],
                ]
                pass
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


def readJsonFile(file: str | Path):
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
