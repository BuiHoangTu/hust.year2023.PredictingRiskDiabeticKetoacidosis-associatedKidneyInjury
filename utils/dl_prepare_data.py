from typing import List

import numpy as np
from pandas import DataFrame, Timedelta
from sklearn.preprocessing import OneHotEncoder
from utils.class_patient import Patients


def trainTest(splitedPatients: List[Patients]):
    for i in range(splitedPatients.__len__()):
        testPatients = splitedPatients[i]

        trainPatientsList = splitedPatients[:i] + splitedPatients[i + 1 :]
        trainPatients = Patients(patients=[])
        for trainPatientsElem in trainPatientsList:
            trainPatients += trainPatientsElem

        yield trainPatients, testPatients


def combineDataframes(dataframes: List[DataFrame]):
    # fill values
    for i in range(1, len(dataframes)):
        dataframes[i].fillna(dataframes[i - 1], inplace=True)
        pass

    # combine dataframes (patients, features, timeWindows)
    arrays = [df.to_numpy() for df in dataframes]
    combinedArray = np.stack(arrays, axis=2)

    # reorder axis (patients, timeWindows, features)
    combinedArray = combinedArray.transpose(0, 2, 1)

    return combinedArray


def get(hoursPerWindows: int):

    def timeWindowGenerate(stop=24):
        start = 0
        while True:
            if start >= stop:
                break

            yield (
                (Timedelta(hours=start) if start > 0 else Timedelta(hours=-6)),
                (
                    Timedelta(hours=(start + hoursPerWindows))
                    if start + hoursPerWindows < stop
                    else Timedelta(hours=stop)
                ),
            )

            start += hoursPerWindows

    patients = Patients()
    if __name__ == "__main__":
        print("retrieved patients", len(patients))

    # fill measures whose null represent false value
    nullableMeasures = [
        "dka_type",
        "macroangiopathy",
        "microangiopathy",
        "mechanical_ventilation",
        "use_NaHCO3",
        "history_aci",
        "history_ami",
        "congestive_heart_failure",
        "liver_disease",
        "ckd_stage",
        "malignant_cancer",
        "hypertension",
        "uti",
        "chronic_pulmonary_disease",
    ]
    for measureName in nullableMeasures:
        patients.fillMissingMeasureValue(measureName, 0)

    # remove measures with less than 80% of data

    measures = patients.getMeasures()

    for measure, count in measures.items():
        if count < len(patients) * 80 / 100:
            patients.removeMeasures([measure])
            # print(measure, count)

    # remove patients with less than 80% of data
    patients.removePatientByMissingFeatures()
    if __name__ == "__main__":
        print("removed patients and features", len(patients))

    splitedPatients = patients.split(5, 27)

    for pTrain, pTest in trainTest(splitedPatients):
        trainLabel = [p.akdPositive for p in pTrain]
        testLabel = [p.akdPositive for p in pTest]

        trainList = []
        testList = []
        for start, stop in timeWindowGenerate():
            categoryColumns = [
                "dka_type",
                "gender",
                "race",
                "liver_disease",
                "ckd_stage",
            ]

            oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

            dfTrain = pTrain.getMeasuresBetween(start, stop).drop(
                columns=["subject_id", "hadm_id", "stay_id", "akd"]
            )
            oneHotEncoder.fit(dfTrain[categoryColumns])
            encoded = oneHotEncoder.transform(
                dfTrain[categoryColumns]
            )
            dfEncoded = DataFrame(
                encoded, # type: ignore
                columns=oneHotEncoder.get_feature_names_out(categoryColumns),
            )
            dfTrain = dfTrain.drop(columns=categoryColumns)
            dfTrain = dfTrain.join(dfEncoded)
            trainList.append(dfTrain)

            dfTest = pTest.getMeasuresBetween(start, stop).drop(
                columns=["subject_id", "hadm_id", "stay_id", "akd"]
            )
            encoded = oneHotEncoder.transform(dfTest[categoryColumns])
            dfEncoded = DataFrame(
                encoded, # type: ignore
                columns=oneHotEncoder.get_feature_names_out(categoryColumns),
            )
            dfTest = dfTest.drop(columns=categoryColumns)
            dfTest = dfTest.join(dfEncoded)
            testList.append(dfTest)
            pass

        train = combineDataframes(trainList)
        test = combineDataframes(testList)

        yield train, trainLabel, test, testLabel

    pass


def timeWindowGenerate(hoursPerWindow=1, stop=24):
    start = 0
    while True:
        if start >= stop:
            break

        yield (
            (Timedelta(hours=start) if start > 0 else Timedelta(hours=-6)),
            (
                Timedelta(hours=(start + hoursPerWindow))
                if start + hoursPerWindow < stop
                else Timedelta(hours=stop)
            ),
        )
