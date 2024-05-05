from typing import Collection, List
from class_patient import Patient, toJsonFile
from constants import TEMP_PATH
from patients import getTargetPatientIcu
import akd_positive
from variables.charateristics_diabetes import (
    getDiabeteType,
    getMacroangiopathy,
    getMicroangiopathy,
)
from variables.demographics import getAge, getEthnicity, getGender, getHeight, getWeight
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


def putDataForPatients(patients: Collection[Patient], df):
    for patient in patients:
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


patientList: List[Patient] = []

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
putDataForPatients(patientList, df)


df = getMacroangiopathy()
putDataForPatients(patientList, df)


df = getMicroangiopathy()
putDataForPatients(patientList, df)


########### Demographics ###########
df = getAge()
putDataForPatients(patientList, df)


df = getGender()
putDataForPatients(patientList, df)


df = getEthnicity()
putDataForPatients(patientList, df)


df = getHeight()
putDataForPatients(patientList, df)


df = getWeight()
putDataForPatients(patientList, df)


########### Laboratory test ###########
df = lab_test.getWbc().dropna()
putDataForPatients(patientList, df)


df = lab_test.getLymphocyte().dropna()
putDataForPatients(patientList, df)


df = lab_test.getHb().dropna()
putDataForPatients(patientList, df)


df = lab_test.getPlt().dropna()
putDataForPatients(patientList, df)


df = lab_test.getPO2().dropna()
putDataForPatients(patientList, df)


df = lab_test.getPCO2().dropna()
putDataForPatients(patientList, df)


df = lab_test.get_pH().dropna()
putDataForPatients(patientList, df)


df = lab_test.getAG().dropna()
putDataForPatients(patientList, df)


df = lab_test.getBicarbonate().dropna()
putDataForPatients(patientList, df)


df = lab_test.getBun().dropna()
putDataForPatients(patientList, df)


df = lab_test.getCalcium().dropna()
putDataForPatients(patientList, df)


df = lab_test.getScr().dropna()
putDataForPatients(patientList, df)


df = lab_test.getBg().dropna()
putDataForPatients(patientList, df)


df = lab_test.getPhosphate().dropna()
putDataForPatients(patientList, df)


df = lab_test.getAlbumin().dropna()
putDataForPatients(patientList, df)


df = lab_test.get_eGFR().dropna()
putDataForPatients(patientList, df)


df = lab_test.getHbA1C().dropna()
putDataForPatients(patientList, df)


df = lab_test.getCrp().dropna()
putDataForPatients(patientList, df)


df = lab_test.getUrineKetone().dropna()
putDataForPatients(patientList, df)


########### Scoring systems ###########
df = getGcs().dropna()
putDataForPatients(patientList, df)


df = getOasis().dropna()
putDataForPatients(patientList, df)


df = getSofa()
putDataForPatients(patientList, df)


df = getSaps2()
putDataForPatients(patientList, df)


########### Vital signs ###########
df = getHeartRate().dropna()
putDataForPatients(patientList, df)


df = getRespiratoryRate().dropna()
putDataForPatients(patientList, df)


df = getSystolicBloodPressure().dropna()
putDataForPatients(patientList, df)


df = getDiastolicBloodPressure().dropna()
putDataForPatients(patientList, df)


########### Prognosis ###########
df = getPreIcuLos().dropna()
putDataForPatients(patientList, df)


df = getHistoryACI()
putDataForPatients(patientList, df)


########### Comorbidities ###########
df = getHistoryAMI()
putDataForPatients(patientList, df)


df = getCHF()
putDataForPatients(patientList, df)


df = getLiverDisease()
putDataForPatients(patientList, df)


df = getPreExistingCKD()
putDataForPatients(patientList, df)


df = getMalignantCancer()
putDataForPatients(patientList, df)


df = getHypertension()
putDataForPatients(patientList, df)


df = getUTI()
putDataForPatients(patientList, df)


df = getChronicPulmonaryDisease()
putDataForPatients(patientList, df)


########### Save file ###########
toJsonFile(patientList, TEMP_PATH / "learning_data.json")