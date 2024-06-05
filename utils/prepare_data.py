from typing import Iterable, List
import numpy as np
from pandas import DataFrame, Timedelta
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from constants import CATEGORICAL_MEASURES
from utils.class_outlier import Outlier
from utils.class_patient import Patients
import pandas as pd


def patientsToNumpy(
    patients: Patients,
    hoursPerWindows: int,
    categoricalColumns: List[str],
    columns: Iterable[str] | None = None,
    categoricalEncoder: None | OneHotEncoder = None,
    numericEncoder: None | StandardScaler = None,
    outlier: Outlier | None = None,
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
    if categoricalEncoder is None:
        categoricalEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    if numericEncoder is None:
        numericEncoder = StandardScaler()

    if outlier is None:
        outlier = Outlier()

    if __name__ == "__main__":
        print("retrieved patients", len(patients))

    dfPatientList: List[DataFrame] = []
    for start, stop in timeWindowGenerate():
        dfPatient = patients.getMeasuresBetween(start, stop).drop(
            columns=["subject_id", "hadm_id", "stay_id", "akd"]
        )
        dfPatientList.append(dfPatient)
        pass

    dfTmp = pd.concat(dfPatientList, axis=0)
    if columns is None:
        numeriColumns = [
            col
            for col in dfTmp.columns
            if col not in categoricalColumns and dfTmp[col].dtype != "bool"
        ]
    else:
        outlierCateCols = categoricalEncoder.get_feature_names_out(categoricalColumns)

        numeriColumns = [
            col
            for col in columns
            if col not in outlierCateCols and dfTmp[col].dtype != "bool"
        ]
    # Outlier
    if outlier.fitted is False:
        outlier.fit(pd.concat(dfPatientList, axis=0)[numeriColumns])

    for i, df in enumerate(dfPatientList):
        dfPatientList[i][numeriColumns] = outlier.transform(df[numeriColumns])

    # fill values
    for i in range(1, len(dfPatientList)):
        dfPatientList[i].fillna(dfPatientList[i - 1], inplace=True)
        pass

    # encode categorical columns
    if (
        not hasattr(categoricalEncoder, "categories_")
        or categoricalEncoder.categories_ is None
    ):
        categoricalEncoder.fit(pd.concat(dfPatientList, axis=0)[categoricalColumns])

    for i, df in enumerate(dfPatientList):
        encoded = categoricalEncoder.transform(df[categoricalColumns])
        dfEncoded = DataFrame(
            encoded,  # type: ignore
            columns=categoricalEncoder.get_feature_names_out(categoricalColumns),
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
        dfAll = pd.concat(dfPatientList, axis=0).astype(np.float32)
        numericEncoder.fit(dfAll)
        columns = dfAll.columns

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
        categoricalEncoder,
        numericEncoder,
        outlier,
        columns,
    )


def normalizeData(dfTrain: DataFrame, dfTest, dfVal=None):
    return _normalizeData(dfTrain, dfTest, dfVal, fillData=False)


def normalizeAndFillData(dfTrain, dfTest, dfVal=None):
    return _normalizeData(dfTrain, dfTest, dfVal, fillData=True)


def encodeCategoricalData(dfTrain: DataFrame, dfTest, dfVal=None):
    return _normalizeData(dfTrain, dfTest, dfVal, encodeNumeric=False)


def _normalizeData(
    dfTrain: DataFrame, dfTest, dfVal=None, fillData=False, encodeCategorical=True, encodeNumeric=True
):
    numericColumns = dfTrain.select_dtypes(include=[np.number]).columns
    numericColumns = [x for x in numericColumns if x not in CATEGORICAL_MEASURES]

    if encodeNumeric:
        # oulier
        outliers = Outlier()
        dfTrain[numericColumns] = outliers.fit_transform(dfTrain[numericColumns])

        # knn
        if fillData:
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            dfTrain[numericColumns] = imputer.fit_transform(dfTrain[numericColumns])

    # category
    if encodeCategorical:
        oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = oneHotEncoder.fit_transform(dfTrain[CATEGORICAL_MEASURES])
        dfEncoded = pd.DataFrame(
            encoded, columns=oneHotEncoder.get_feature_names_out(CATEGORICAL_MEASURES)
        )
        dfTrain = dfTrain.drop(columns=CATEGORICAL_MEASURES)
        dfTrain = dfTrain.join(dfEncoded)

    # numeric
    if encodeNumeric:
        standardScaler = StandardScaler()
        dfTrain[numericColumns] = standardScaler.fit_transform(dfTrain[numericColumns])

    # matching columns
    columns = dfTrain.columns

    if encodeNumeric:
        # outlier
        dfTest[numericColumns] = outliers.transform(dfTest[numericColumns])

        # knn
        if fillData:
            dfTest[numericColumns] = imputer.transform(dfTest[numericColumns])

    # category
    if encodeCategorical:
        encoded = oneHotEncoder.transform(dfTest[CATEGORICAL_MEASURES])
        dfEncoded = pd.DataFrame(encoded, columns=oneHotEncoder.get_feature_names_out(CATEGORICAL_MEASURES))  # type: ignore
        dfTest = dfTest.drop(columns=CATEGORICAL_MEASURES)
        dfTest = dfTest.join(dfEncoded)

    # numeric
    if encodeNumeric:
        dfTest[numericColumns] = standardScaler.transform(dfTest[numericColumns])

    # matching columns
    for col in columns:
        if col not in dfTest.columns:
            dfTest[col] = np.nan
            pass
        pass
    dfTest = dfTest[columns]

    if dfVal is None:
        return dfTrain, dfTest, None

    if encodeNumeric:
        # outlier
        dfVal[numericColumns] = outliers.transform(dfVal[numericColumns])

        # knn
        if fillData:
            dfVal[numericColumns] = imputer.transform(dfVal[numericColumns])

    # category
    if encodeCategorical:
        encoded = oneHotEncoder.transform(dfVal[CATEGORICAL_MEASURES])
        dfEncoded = pd.DataFrame(encoded, columns=oneHotEncoder.get_feature_names_out(CATEGORICAL_MEASURES))  # type: ignore
        dfVal = dfVal.drop(columns=CATEGORICAL_MEASURES)
        dfVal = dfVal.join(dfEncoded)

    # numeric
    if encodeNumeric:
        dfVal[numericColumns] = standardScaler.transform(dfVal[numericColumns])

    # matching columns
    for col in columns:
        if col not in dfVal.columns:
            dfVal[col] = np.nan
            pass
        pass

    return dfTrain, dfTest, dfVal
