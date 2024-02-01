import pandas as pd
from constants import IMPORTANT_MESUREMENTS_ICU, PREPROCESSED_MESUREMENTS, TEMP_PATH


def mergeData(dfPatient: pd.DataFrame, mesureInterval: pd.Timedelta, mesureGroupLength: int):
    """Run this function to merge all chartevent csv data in TEMP_PATH with patients

    Args:
        dfPatient (pd.DataFrame): patients' data, including these field: stay_id, intime, outtime 
    """
    
    ###################################### Function start ###################################### 

    dfWillAkd = pd.read_csv(TEMP_PATH / "will_akd.csv", parse_dates=["charttime"])
    dfs = []
    
    mesuresName = PREPROCESSED_MESUREMENTS + list(IMPORTANT_MESUREMENTS_ICU.values())
    mesuresName.extend(PREPROCESSED_MESUREMENTS)
    mesureMap = dict()
    for mesureName in mesuresName:
        dfMesure = pd.read_csv(TEMP_PATH / ("chartevent_" + mesureName + ".csv"), parse_dates=["charttime"])
        mesureMap[mesureName] = dfMesure
        pass
    
    for rowId, row in dfPatient.iterrows():
        for i in range(99):
            if (row["intime"] + (i + mesureGroupLength) * mesureInterval > row["outtime"]):
                break

            # get last group, check will_akd
            newDf = pd.DataFrame(data=row, copy=True)
            lastFrameTime = row["intime"] + (i + mesureGroupLength) * mesureInterval
            currentPatientWillAkd = dfWillAkd[dfWillAkd["stay_id"] == row["stay_id"]]
            willAkd = any(
                (lastFrameTime <= currentPatientWillAkd["charttime"]) &
                (currentPatientWillAkd["charttime"] < lastFrameTime + mesureInterval) &
                (dfWillAkd["will_akd"])
            )
            newDf["will_akd"] = willAkd
            
            # find mesures 
            
            for mesureName in mesuresName:
                dfCurrentMesure: pd.DataFrame = mesureMap[mesureName]
                dfCurrentMesure = dfCurrentMesure[dfCurrentMesure["stay_id"] == row["stay_id"]]
                
                mesureValues = []
                for ii in range(mesureGroupLength):
                    lowTime = row["intime"] + (i + ii) * mesureInterval
                    highTime = lowTime + mesureInterval
                    
                    iiMesures = dfCurrentMesure[
                        (lowTime <= dfCurrentMesure["charttime"]) &
                        (dfCurrentMesure["charttime"] < highTime)                    
                    ]
                    
                    if (len(iiMesures) <= 0):
                        value = 0
                        pass
                    else:
                        value = iiMesures["valuenum"].std()
                        pass
                    
                    mesureValues.append(value)
                    pass
                
                for idx, value in enumerate(mesureValues):
                    newDf[mesureName + str(idx)] = value
                    pass    
                pass
            
            dfs.append(newDf)
            pass
        pass

    mainDf = pd.concat(dfs)
    mainDf.to_csv(TEMP_PATH / "maindf.csv")
    
    pass

if __name__ == "__main__":
    dfPatient = pd.read_csv(TEMP_PATH / "dfPatient.csv", parse_dates=["intime","outtime"])
    
    MESURE_INTERVAL = pd.Timedelta(hours=6)
    MESURE_GROUP_LENGTH = 6

    mergeData(dfPatient, MESURE_INTERVAL, MESURE_GROUP_LENGTH)
    pass