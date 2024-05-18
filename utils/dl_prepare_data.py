from typing import Iterable, List
import numpy as np
from pandas import DataFrame, Timedelta
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
import pandas as pd


def patientsToNumpy(
    patients: Patients,
    hoursPerWindows: int,
    categoricalColumns: List[str],
    columns: Iterable[str] | None = None,
    oneHotEncoder: None | OneHotEncoder = None,
    numericEncoder: None | StandardScaler = None,
):
    """Convert patients to 3d numpy array

    Args:
        patients (Patients): patients
        hoursPerWindows (int): _description_
        oneHotEncoder (None | OneHotEncoder): how to encode categorical columns, if it is not fitted yet, it will be fitted.
        categoricalColumns (List[str]): categorical columns
        numericEncoder (None | StandardScaler): how to encode numeric columns, if it is not fitted yet, it will be fitted.

    Returns:
        np.array: 3d numpy array
        oneHotEncoder: oneHotEncoder(fitted) to encode the test part
        numericEncoder: numericEncoder(fitted) to encode the test part
    """

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

    dfPatientList: List[DataFrame] = []
    for start, stop in timeWindowGenerate():
        dfPatient = patients.getMeasuresBetween(start, stop).drop(
            columns=["subject_id", "hadm_id", "stay_id", "akd"]
        )
        dfPatientList.append(dfPatient)
        pass

    # fill values
    for i in range(1, len(dfPatientList)):
        dfPatientList[i].fillna(dfPatientList[i - 1], inplace=True)
        pass

    # encode categorical columns
    if not hasattr(oneHotEncoder, "categories_") or oneHotEncoder.categories_ is None:
        oneHotEncoder.fit(pd.concat(dfPatientList, axis=0)[categoricalColumns])

    for i, df in enumerate(dfPatientList):
        encoded = oneHotEncoder.transform(df[categoricalColumns])
        dfEncoded = DataFrame(
            encoded,  # type: ignore
            columns=oneHotEncoder.get_feature_names_out(categoricalColumns),
        )

        # replace original columns with encoded columns
        dfMerged = df.drop(columns=categoricalColumns)
        dfMerged = dfMerged.join(dfEncoded)
        dfPatientList[i] = dfMerged
        pass

    # ensure columns order for numeric encode (for test set)
    if columns is not None:
        for i, df in enumerate(dfPatientList):
            for col in columns:
                if col not in df.columns:
                    df[col] = np.nan
            pass

        dfPatientList[i] = df[columns]

    # encode numeric values
    if (not hasattr(numericEncoder, "mean_") or numericEncoder.mean_ is None) and (
        not hasattr(numericEncoder, "scale_") or numericEncoder.scale_ is None
    ):
        numericEncoder.fit(pd.concat(dfPatientList, axis=0).astype(np.float32))

    for i, df in enumerate(dfPatientList):
        encoded = numericEncoder.transform(df.astype(np.float32))
        dfEncoded = DataFrame(
            encoded,  # type: ignore
            columns=df.columns,
        )

        # replace original columns with encoded columns
        dfPatientList[i] = dfEncoded
        pass

    # combine dataframes (patients, features, timeWindows)
    arrays = [df.to_numpy(dtype=np.float32) for df in dfPatientList]
    combinedArray = np.stack(arrays, axis=2)

    # reorder axis (patients, timeWindows, features)
    npPatient = combinedArray.transpose(0, 2, 1)

    return (
        npPatient,
        oneHotEncoder,
        numericEncoder,
        pd.concat(dfPatientList, axis=0).columns,
    )
