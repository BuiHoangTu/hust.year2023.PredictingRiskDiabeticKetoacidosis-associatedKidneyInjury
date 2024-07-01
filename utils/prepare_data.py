from typing import Iterable, List
import numpy as np
from pandas import DataFrame, Timedelta
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from constants import CATEGORICAL_MEASURES
from utils.class_outlier import Outliner
from utils.class_patient import Patient, Patients
import pandas as pd


def patientsToNumpy(
    patients: Patients,
    hoursPerWindows: int,
    categoricalColumns: List[str],
    columns: Iterable[str] | None = None,
    categoricalEncoder: None | OneHotEncoder = None,
    numericEncoder: None | StandardScaler = None,
    outlier: Outliner | None = None,
    timeSeriesOnly: bool = False,
    fromHour: int = 0,
    toHour: int = 24,
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

    if timeSeriesOnly:
        measureTypes = "time"
    else:
        measureTypes = "all"

    def timeWindowGenerate():
        start = fromHour
        stop = toHour
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
        outlier = Outliner()

    if __name__ == "__main__":
        print("retrieved patients", len(patients))

    dfPatientList: List[DataFrame] = []
    for start, stop in timeWindowGenerate():
        dfPatient = patients.getMeasuresBetween(
            start, stop, measureTypes=measureTypes
        ).drop(columns=["subject_id", "hadm_id", "stay_id", "akd"])
        dfPatientList.append(dfPatient)
        pass

    dfTmp = pd.concat(dfPatientList, axis=0)
    if columns is None:
        categoricalColumns = [col for col in dfTmp.columns if col in categoricalColumns]

        numeriColumns = [
            col
            for col in dfTmp.columns
            if col not in categoricalColumns and dfTmp[col].dtype != "bool"
        ]
    else:
        categoricalColumns = [col for col in columns if col in categoricalColumns]

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
    dfTrain: DataFrame,
    dfTest,
    dfVal=None,
    fillData=False,
    encodeCategorical=True,
    encodeNumeric=True,
):
    normalizeData = NormalizeData(fillData, encodeCategorical, encodeNumeric)

    dfTrain = normalizeData.fit_transform(dfTrain)
    dfTest = normalizeData.transform(dfTest)
    if dfVal is not None:
        dfVal = normalizeData.transform(dfVal)

    return dfTrain, dfTest, dfVal


class NormalizeData:
    def __init__(self, fillData=False, encodeCategorical=True, encodeNumeric=True):
        self.fillData = fillData
        self.encodeCategorical = encodeCategorical
        self.encodeNumeric = encodeNumeric

        self.oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.standardScaler = StandardScaler()
        self.outliers = Outliner()
        self.columns: Iterable[str] = []
        self.imputer = KNNImputer(n_neighbors=6, weights="distance")
        self.numericColumns: Iterable[str] = []
        self.categoricalColumns: Iterable[str] = []
        self.mixedColumns: Iterable[str] = []

    def fit(self, dfTrain: DataFrame):
        self.fit_transform(dfTrain)

    def fit_transform(self, dfTrain: DataFrame):
        self.numericColumns = dfTrain.select_dtypes(include=[np.number]).columns
        self.numericColumns = [
            x for x in self.numericColumns if x not in CATEGORICAL_MEASURES
        ]

        self.categoricalColumns = [
            x for x in CATEGORICAL_MEASURES if x in dfTrain.columns
        ]
        self.mixedColumns = [
            x for x in dfTrain.columns if x not in CATEGORICAL_MEASURES
        ]

        if self.encodeNumeric:
            # oulier
            dfTrain[self.numericColumns] = self.outliers.fit_transform(
                dfTrain[self.numericColumns]
            )

            # knn
            dfTrain[self.numericColumns] = self.imputer.fit_transform(
                dfTrain[self.numericColumns]
            )

        # category
        if self.encodeCategorical:
            encoded = self.oneHotEncoder.fit_transform(dfTrain[self.categoricalColumns])
            dfEncoded = pd.DataFrame(
                encoded,
                columns=self.oneHotEncoder.get_feature_names_out(
                    self.categoricalColumns
                ),
            )
            dfTrain = dfTrain.drop(columns=self.categoricalColumns)
            dfTrain = dfTrain.join(dfEncoded)

        # parse mixed to number
        for col in self.mixedColumns:
            dfTrain[col] = dfTrain[col].astype(float)

        # numeric
        if self.encodeNumeric:
            dfTrain[self.numericColumns] = self.standardScaler.fit_transform(
                dfTrain[self.numericColumns]
            )

        # matching columns
        self.columns = dfTrain.columns

        return dfTrain

    def transform(self, df: DataFrame):
        if self.encodeNumeric:
            # outlier
            df[self.numericColumns] = self.outliers.transform(df[self.numericColumns])

            # knn
            if self.fillData:
                df[self.numericColumns] = self.imputer.transform(
                    df[self.numericColumns]
                )

        # category
        if self.encodeCategorical:
            encoded = self.oneHotEncoder.transform(df[self.categoricalColumns])
            dfEncoded = pd.DataFrame(
                encoded, columns=self.oneHotEncoder.get_feature_names_out(list(self.categoricalColumns))  # type: ignore
            )
            df = df.drop(columns=self.categoricalColumns)
            df = df.join(dfEncoded)

        # parse mixed to number
        for col in self.mixedColumns:
            df[col] = df[col].astype(float)

        # numeric
        if self.encodeNumeric:
            df[self.numericColumns] = self.standardScaler.transform(
                df[self.numericColumns]
            )

        # matching columns
        for col in self.columns:
            if col not in df.columns:
                df[col] = np.nan
                pass
            pass
        df = df[self.columns]

        return df


class DLModel:
    def __init__(
        self,
        model,
        normalizeData: NormalizeData,
        categoryEncoder,
        numericEncoder,
        oulier,
        columns,
        hoursPerWindows,
        fromHour,
        toHour,
    ):
        self.model = model
        self.normalizeData = normalizeData
        self.categoryEncoder = categoryEncoder
        self.numericEncoder = numericEncoder
        self.oulier = oulier
        self.columns = columns
        self.hoursPerWindows = hoursPerWindows
        self.fromHour = fromHour
        self.toHour = toHour

        pass
    
    def predict(self, patient: Patient):
        npX = patientsToNumpy(
            Patients([patient]),
            self.hoursPerWindows,
            CATEGORICAL_MEASURES,
            self.columns,
            self.categoryEncoder,
            self.numericEncoder,
            self.oulier,
            timeSeriesOnly=True,
            fromHour=self.fromHour,
            toHour=self.toHour,
        )[0]
        npX = np.nan_to_num(npX, nan=0)
        
        staticX = patient.getMeasuresBetween(measureTypes="static")
        staticX = staticX.drop(columns=["subject_id", "hadm_id", "stay_id", "akd"])
        staticX = self.normalizeData.transform(staticX)
        staticX = staticX.to_numpy(dtype=np.float32)
        staticX = np.nan_to_num(staticX, nan=0)
        
        
        return self.model.predict([npX, staticX])
