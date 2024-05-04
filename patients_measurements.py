from typing import List

from class_patient import Patient, toJsonFile


patient_list: List[Patient] = []


def putDataForPatients(patient_list, df):
    for patient in patient_list:
        if "stay_id" in df.columns:
            filtered_df = df[df["stay_id"] == patient.stayId]
        elif "hadm_id" in df.columns:
            filtered_df = df[df["hadm_id"] == patient.hadmId]
        elif "subject_id" in df.columns:
            filtered_df = df[df["subject_id"] == patient.subjectId]
        else:
            print("DataFrame does not have 'hadm_id' or 'stay_id' column.")
            return

        filtered_df = filtered_df.reset_index(drop=True)

        dataColumns = [
            x for x in filtered_df.columns if x not in ["stay_id", "hadm_id", "time"]
        ]

        for _, row in filtered_df.iterrows():
            for dataColumn in dataColumns:
                patient.putMeasure(dataColumn, row.get("time"), row[dataColumn])


from constants import TEMP_PATH
from patients import getTargetPatientIcu


dfPatient = getTargetPatientIcu()
dfPatient = dfPatient[["subject_id", "hadm_id", "stay_id"]]

dfPatient


import akd_positive


dfAkd = akd_positive.extractKdigoStages7day()
dfAkd["akd"] = dfAkd["aki_7day"]
dfAkd = dfAkd[["stay_id", "akd"]]

dfData1 = dfPatient.merge(dfAkd, "left", "stay_id")
dfData1["akd"] = dfData1["akd"].astype(bool)

for _, row in dfData1.iterrows():
    patient = Patient(row["subject_id"], row["hadm_id"], row["stay_id"], row["akd"])
    patient_list.append(patient)
    pass

dfData1["akd"].value_counts()


from variables_charateristics_diabetes import getDiabeteType


df = getDiabeteType()
df["dka_type"] = df["dka_type"].astype(int)
putDataForPatients(patient_list, df)


import variables_charateristics_diabetes


df = variables_charateristics_diabetes.getMacroangiopathy()
putDataForPatients(patient_list, df)


import variables_charateristics_diabetes


df = variables_charateristics_diabetes.getMicroangiopathy()
putDataForPatients(patient_list, df)


import variables_demographics


df = variables_demographics.getAge()
df.__len__()


from matplotlib import pyplot as plt


putDataForPatients(patient_list, df)


df = variables_demographics.getGender()
df.__len__()


putDataForPatients(patient_list, df)


df = variables_demographics.getEthnicity()
df.__len__()


putDataForPatients(patient_list, df)


df = variables_demographics.getHeight()
df.__len__()


from variables_demographics import getWeight


df = getWeight()


df = df.drop_duplicates("stay_id", keep="first")

df.__len__()


from matplotlib import pyplot as plt

putDataForPatients(patient_list, df)


import variables_interventions


putDataForPatients(patient_list, df)


putDataForPatients(patient_list, df)


putDataForPatients(patient_list, df)


import variables_lab_test

df = variables_lab_test.getWbc().dropna()
df


from matplotlib import pyplot as plt

putDataForPatients(patient_list, df)

plt.hist(df["wbc"], bins=30, edgecolor="black")
plt.show()


df = variables_lab_test.getLymphocyte().dropna()
df.__len__()


df = variables_lab_test.getHb().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["hb"], bins=30, edgecolor="black")
plt.show()

putDataForPatients(patient_list, df)


df = variables_lab_test.getPlt().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["plt"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getPO2().dropna()
df.__len__()


df = variables_lab_test.getPCO2().dropna()
df.__len__()


df = variables_lab_test.get_pH().dropna()
df.__len__()


df = variables_lab_test.getAG().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["ag"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getBicarbonate().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["bicarbonate"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getBun().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["bun"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getCalcium().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["calcium"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getScr().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["scr"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getBg().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["bg"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getPhosphate().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["phosphate"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getAlbumin().dropna()
df.__len__()


df = variables_lab_test.get_eGFR().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["egfr"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


df = variables_lab_test.getHbA1C().dropna()
df.__len__()


df = variables_lab_test.getCrp().dropna()
df.__len__()


df = variables_lab_test.getUrineKetone().dropna()
df.__len__()


import variables_scoring_systems


df = variables_scoring_systems.getGcs().dropna()
df.__len__()


putDataForPatients(patient_list, df)
df["gcs"].value_counts()


from variables_scoring_systems import getOasis


df = getOasis().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["oasis"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


from variables_scoring_systems import getSofa


df = getSofa()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["sofa"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


from variables_scoring_systems import getSaps2


df = getSaps2()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["saps2"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


from reduce_mesurements import reduceByStayId
from variables_vital_signs import getHeartRate


df = getHeartRate().dropna()


df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["hr"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


from variables_vital_signs import getRespiratoryRate


df = getRespiratoryRate().dropna()

df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["rr"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


from variables_vital_signs import getSystolicBloodPressure


df = getSystolicBloodPressure().dropna()

df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["sbp"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


from variables_vital_signs import getDiastolicBloodPressure


df = getDiastolicBloodPressure().dropna()

df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["dbp"], bins=30, edgecolor="black")
plt.show()
putDataForPatients(patient_list, df)


from variables_prognosis import getPreIcuLos


df = getPreIcuLos().dropna()
df.__len__()


from matplotlib import pyplot as plt

plt.hist(df["preiculos"], bins=30, edgecolor="black")
plt.show()


from matplotlib import pyplot as plt
import numpy as np

LOS_FLOOR = 365
preiculos = np.where(df["preiculos"] > LOS_FLOOR, LOS_FLOOR, df["preiculos"])

plt.hist(preiculos, bins=30, edgecolor="black")
plt.show()


df["preiculos"] = preiculos
putDataForPatients(patient_list, df)


from variables_comorbidities import getHistoryACI


df = getHistoryACI()
putDataForPatients(patient_list, df)


from variables_comorbidities import getHistoryAMI


df = getHistoryAMI()
putDataForPatients(patient_list, df)


from variables_comorbidities import getCHF


df = getCHF()
putDataForPatients(patient_list, df)


from variables_comorbidities import getLiverDisease


df = getLiverDisease()
putDataForPatients(patient_list, df)


from variables_comorbidities import getPreExistingCKD


df = getPreExistingCKD()
putDataForPatients(patient_list, df)


from variables_comorbidities import getMalignantCancer


df = getMalignantCancer()
putDataForPatients(patient_list, df)


from variables_comorbidities import getHypertension


df = getHypertension()
putDataForPatients(patient_list, df)


from variables_comorbidities import getUTI


df = getUTI()
putDataForPatients(patient_list, df)


from variables_comorbidities import getChronicPulmonaryDisease


df = getChronicPulmonaryDisease()
putDataForPatients(patient_list, df)


toJsonFile(patient_list, TEMP_PATH / "learning_data.json")
