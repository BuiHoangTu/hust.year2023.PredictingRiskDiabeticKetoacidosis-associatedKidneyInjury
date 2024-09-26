from pathlib import Path
import pickle
from typing import Iterable, List
import numpy as np
from pandas import DataFrame, Timedelta
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from constants import CATEGORICAL_MEASURES, NULLABLE_MEASURES, TEMP_PATH
from utils.class_outlier import Outliner
from utils.class_patient import Patient, Patients
import pandas as pd


class DataNormalizer:
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
        return self

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

            if self.fillData:
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

        # parse mixed to number if exist in df
        for col in set(self.mixedColumns) & set(df.columns):
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


# class DLModel:
#     def __init__(
#         self,
#         model,
#         normalizeData: DataNormalizer,
#         hoursPerWindows,
#         fromHour,
#         toHour,
#     ):
#         self.model = model
#         self.normalizeData = normalizeData
#         self.hoursPerWindows = hoursPerWindows
#         self.fromHour = fromHour
#         self.toHour = toHour

#         pass

#     def predict(self, patient: Patient):
#         npX = patientsToNumpy(
#             Patients([patient]),
#             self.hoursPerWindows,
#             timeSeriesOnly=True,
#             fromHour=self.fromHour,
#             toHour=self.toHour,
#             dataNormalizer=self.normalizeData,
#             isTrainPatients=False,
#         )[0]
#         npX = np.nan_to_num(npX, nan=0)

#         staticX = patient.getMeasuresBetween(measureTypes="static")
#         staticX = staticX.drop(columns=["subject_id", "hadm_id", "stay_id", "akd"])
#         staticX = self.normalizeData.transform(staticX)
#         staticX = staticX.to_numpy(dtype=np.float32)
#         staticX = np.nan_to_num(staticX, nan=0)

#         return self.model.predict([npX, staticX])


def patientsToNumpy(
    patients: Patients,
    hoursPerWindows: int,
    timeSeriesOnly: bool = False,
    fromHour: int = 0,
    toHour: int = 24,
    dataNormalizer: DataNormalizer | None = None,
    isTrainPatients: bool = True,
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

    if not isTrainPatients and dataNormalizer is None:
        raise ValueError(
            "normalizeDataCls must be provided if this is not TrainPatients"
        )

    converter = PatientsNumpyConverter(
        hoursPerWindows=hoursPerWindows,
        timeSeriesOnly=timeSeriesOnly,
        fromHour=fromHour,
        toHour=toHour,
    )    
    if dataNormalizer is not None:
        converter.dataNormalizer = DataNormalizer()

    if __name__ == "__main__":
        print("retrieved patients", len(patients))

    if isTrainPatients:
        combinedArray = converter.fit_transform(patients)
    else:
        combinedArray = converter.transform(patients)

    return (combinedArray, converter.dataNormalizer)


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
    normalizeData = DataNormalizer(fillData, encodeCategorical, encodeNumeric)

    dfTrain = normalizeData.fit_transform(dfTrain)
    dfTest = normalizeData.transform(dfTest)
    if dfVal is not None:
        dfVal = normalizeData.transform(dfVal)

    return dfTrain, dfTest, dfVal


def getMonitoredPatients():
    patients = Patients.loadPatients()
    patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

    measures = patients.getMeasures()
    for measure, count in measures.items():
        if count < len(patients) * 80 / 100:
            patients.removeMeasures([measure])

    patients.removePatientByMissingFeatures()

    return patients


def getTimeMonitoredPatients():
    patients = getMonitoredPatients()
    patients.removePatientAkiEarly(Timedelta(hours=12))

    return patients


def trainTestPatients(patients: Patients, seed=27):
    splitedPatients = patients.split(5, seed)

    for i in range(splitedPatients.__len__()):
        testPatients = splitedPatients[i]

        trainPatientsList = splitedPatients[:i] + splitedPatients[i + 1 :]
        trainPatients = Patients(patients=[])
        for trainPatientsElem in trainPatientsList:
            trainPatients += trainPatientsElem

        yield trainPatients, testPatients


def trainValTestPatients(patients: Patients, seed=27):
    splitedTestPatients = patients.split(5, seed)

    for i in range(splitedTestPatients.__len__()):
        testPatients = splitedTestPatients[i]

        trainPatientsList = splitedTestPatients[:i] + splitedTestPatients[i + 1 :]
        trainPatients = Patients(patients=[])
        for trainPatientsElem in trainPatientsList:
            trainPatients += trainPatientsElem

        def trainValGenerator(trainValPatients):
            splitedValPatients = trainValPatients.split(5, seed)
            for ii in range(splitedValPatients.__len__()):
                valPatients = splitedValPatients[ii]
                trainPatientsList = splitedValPatients[:ii] + splitedValPatients[ii + 1 :]
                trainPatients = Patients(patients=[])
                for trainPatientsElem in trainPatientsList:
                    trainPatients += trainPatientsElem
                
                yield trainPatients, valPatients
            pass

        yield trainValGenerator(trainPatients), testPatients


def trainValTestNp(patients: Patients, seed=27, hoursPerWindow=24):
    cacheFile = Path(TEMP_PATH / f"trainTest-{hash(patients)}-{seed}.pkl")

    if cacheFile.exists():
        return pickle.loads(cacheFile.read_bytes())

    output: list[
        tuple[
            int,
            tuple[np.ndarray, np.ndarray, list[bool]],
            tuple[np.ndarray, np.ndarray, list[bool]],
            tuple[np.ndarray, np.ndarray, list[bool]],
        ]
    ] = []
    for i, (trainValGenerator, testPatients) in enumerate(trainValTestPatients(patients, seed)):
        for trainPatients, valPatients in trainValGenerator:
            preparer = DeepLearningDataPreparer(
                hoursPerWindows=hoursPerWindow,
                fromHour=0,
                toHour=12,
            )
        
            npTrainX, staticTrainX, trainY = preparer.fit_transform(trainPatients)
            npValX, staticValX, valY = preparer.transform(valPatients)
            npTestX, staticTestX, testY = preparer.transform(testPatients)
        
            output.append((
                i, 
                (npTrainX, staticTrainX, trainY),
                (npValX, staticValX, valY),
                (npTestX, staticTestX, testY)
            ))
            pass

    cacheFile.write_bytes(pickle.dumps(output))
    return output

class PatientsNumpyConverter():
    def __init__(
        self, 
        hoursPerWindows: int = 24,
        timeSeriesOnly: bool = False,
        fromHour: int = 0,
        toHour: int = 24,
    ):
        self.hoursPerWindows = hoursPerWindows
        if timeSeriesOnly:
            self.measureTypes = "time"
        else:
            self.measureTypes = "all"
        self.fromHour = fromHour
        self.toHour = toHour
        self.dataNormalizer = DataNormalizer()

    def fit(self, patients: Patients):
        dfPatientList: List[DataFrame] = []
        for start, stop in self._timeWindowGenerate():
            dfPatient = patients.getMeasuresBetween(
                start, stop, measureTypes=self.measureTypes # type: ignore
            ).drop(columns=["subject_id", "hadm_id", "stay_id", "akd"])
            dfPatientList.append(dfPatient)
            pass

        dfTmp = pd.concat(dfPatientList, axis=0)
        self.dataNormalizer.fit(dfTmp)

        return self

    def transform(self, patients: Patients):
        dfPatientList: List[DataFrame] = []
        for start, stop in self._timeWindowGenerate():
            dfPatient = patients.getMeasuresBetween(
                start, stop, measureTypes=self.measureTypes  # type: ignore
            ).drop(columns=["subject_id", "hadm_id", "stay_id", "akd"])
            dfPatientList.append(dfPatient)
            pass

        # fill values
        for i in range(1, len(dfPatientList)):
            dfPatientList[i].fillna(dfPatientList[i - 1], inplace=True)
            pass

        for i, df in enumerate(dfPatientList):
            dfEncoded = self.dataNormalizer.transform(df)
            dfPatientList[i] = dfEncoded
            pass

        arrays = [df.to_numpy(dtype=np.float32) for df in dfPatientList]
        combinedArray = np.stack(arrays, axis=1)
        
        return combinedArray

    def fit_transform(self, patients: Patients):
        self.fit(patients)
        return self.transform(patients)

    def _timeWindowGenerate(self):
        start = self.fromHour
        stop = self.toHour
        while True:
            if start >= stop:
                break

            yield (
                (Timedelta(hours=start) if start > 0 else Timedelta(hours=-6)),
                (
                    Timedelta(hours=(start + self.hoursPerWindows))
                    if start + self.hoursPerWindows < stop
                    else Timedelta(hours=stop)
                ),
            )

            start += self.hoursPerWindows
        pass

    pass

class DeepLearningDataPreparer():
    def __init__(
        self, 
        hoursPerWindows,
        fromHour: int = 0,
        toHour: int = 24,
    ):
        self.timeSeriesConverter = PatientsNumpyConverter(
            hoursPerWindows=hoursPerWindows,
            timeSeriesOnly=True,
            fromHour=fromHour,
            toHour=toHour
        )
        
        self.staticNormalizer = DataNormalizer(
            fillData=False,
            encodeCategorical=True,
            encodeNumeric=True
        )
        
    def fit(self, patients: Patients):
        self.timeSeriesConverter.fit(patients)
        
        staticData = self._getRawStatic(patients)
        self.staticNormalizer.fit(staticData)
        return self

    def transform(self, patients: Patients):
        npTimeSeries = self.timeSeriesConverter.transform(patients)
        npTimeSeries = np.nan_to_num(npTimeSeries, nan=0)

        staticData = self._getRawStatic(patients)
        staticData = self.staticNormalizer.transform(staticData)
        staticData = staticData.to_numpy(dtype=np.float32)
        staticData = np.nan_to_num(staticData, nan=0)
        
        labels = [p.akdPositive for p in patients]

        return npTimeSeries, staticData, labels
    
    def fit_transform(self, patients: Patients):
        self.fit(patients)
        return self.transform(patients)
    
    def _getRawStatic(self, patients: Patients):
        staticPatients = patients.getMeasuresBetween(measureTypes="static")
        staticPatients = staticPatients.drop(columns=["subject_id", "hadm_id", "stay_id", "akd"])
        return staticPatients

    pass
