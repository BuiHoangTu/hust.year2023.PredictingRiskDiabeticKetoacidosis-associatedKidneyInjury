from typing import List
import numpy as np
from pandas import DataFrame, Timedelta
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients


def patientsToNumpy(
    patients: Patients,
    hoursPerWindows: int,
    oneHotEncoder: None | OneHotEncoder,
    categoricalColumns: List[str],
    numericEncoder: None | StandardScaler,
):

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
        pass

    # unify inputs
    if oneHotEncoder is None:
        oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    if numericEncoder is None:
        numericEncoder = StandardScaler()

    if __name__ == "__main__":
        print("retrieved patients", len(patients))

    # fill measures whose null represent false value
    for measureName in NULLABLE_MEASURES:
        patients.fillMissingMeasureValue(measureName, 0)

    # # remove measures with less than 80% of data
    # measures = patients.getMeasures()

    # for measure, count in measures.items():
    #     if count < len(patients) * 80 / 100:
    #         patients.removeMeasures([measure])
    #         if __name__ == "__main__":
    #             print("removed", measure, count)

    # # remove patients with less than 80% of data
    # patients.removePatientByMissingFeatures()
    # if __name__ == "__main__":
    #     print("removed patients and features", len(patients))

    dfPatientList = []
    for start, stop in timeWindowGenerate():
        dfPatient = patients.getMeasuresBetween(start, stop).drop(
            columns=["subject_id", "hadm_id", "stay_id", "akd"]
        )

        if (
            not hasattr(oneHotEncoder, "categories_")
            or oneHotEncoder.categories_ is None
        ):
            oneHotEncoder.fit(dfPatient[categoricalColumns])

        encoded = oneHotEncoder.transform(dfPatient[categoricalColumns])
        dfEncoded = DataFrame(
            encoded,  # type: ignore
            columns=oneHotEncoder.get_feature_names_out(categoricalColumns),
        )

        dfPatient = dfPatient.drop(columns=categoricalColumns)
        dfPatient = dfPatient.join(dfEncoded)
        dfPatientList.append(dfPatient)

        pass

    # fill values
    for i in range(1, len(dfPatientList)):
        dfPatientList[i].fillna(dfPatientList[i - 1], inplace=True)
        pass

    # combine dataframes (patients, features, timeWindows)
    arrays = [df.to_numpy() for df in dfPatientList]
    combinedArray = np.stack(arrays, axis=2)

    # reorder axis (patients, timeWindows, features)
    npPatient = combinedArray.transpose(0, 2, 1)

    # scale numeric values
    if (not hasattr(numericEncoder, "mean_") or numericEncoder.mean_ is None) and (
        not hasattr(numericEncoder, "scale_") or numericEncoder.scale_ is None
    ):
        numericEncoder.fit(npPatient[:, :, -1])

    npPatient[:, :, -1] = numericEncoder.transform(npPatient[:, :, -1])

    return npPatient, oneHotEncoder, numericEncoder
