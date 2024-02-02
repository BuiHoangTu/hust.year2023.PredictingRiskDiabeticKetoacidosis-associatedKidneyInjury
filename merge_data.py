import pandas as pd
from constants import IMPORTANT_MESUREMENTS_ICU, PREPROCESSED_MESUREMENTS, TEMP_PATH


def mergeData(dfPatient: pd.DataFrame, mesureInterval: pd.Timedelta, mesureGroupLength: int):
    """Run this function to merge all chartevent csv data in TEMP_PATH with patients

    Args:
        dfPatient (pd.DataFrame): patients' data, including these field: stay_id, intime, outtime 
    """
    
    ###################################### Function start ###################################### 

    dfWillAkd = pd.read_csv(TEMP_PATH / "will_akd.csv", parse_dates=["charttime"])
    targetCsvColumns = list(dfPatient.columns)
    targetCsvColumns.append("will_akd")
    
    mesuresName = PREPROCESSED_MESUREMENTS + list(IMPORTANT_MESUREMENTS_ICU.values())
    mesureMap = dict()
    for mesureName in mesuresName:
        dfMesure = pd.read_csv(TEMP_PATH / ("chartevent_" + mesureName + ".csv"), parse_dates=["charttime"])
        mesureMap[mesureName] = dfMesure
        
        # prepare csv.columns to append 
        for i in range(mesureGroupLength):
            targetCsvColumns.append(mesureName + str(i))
        pass
    dfTargetPlaceholder = pd.DataFrame(columns=targetCsvColumns)
    targetCsvPath = TEMP_PATH / "dfMergedData.csv"
    dfTargetPlaceholder.to_csv(targetCsvPath, index=False)
    
    for rowId, row in dfPatient.iterrows():
        for i in range(99):
            startTime = row["intime"] + i * mesureInterval
            endTime = startTime + mesureGroupLength * mesureInterval # from start -> end: data for prediction
            nextTime = endTime + mesureInterval # from end -> next: time checking for akd 
            
            # check if patient is still in icu 
            if (nextTime > row["outtime"]):
                break
            
            # check if this iteration contains is_akd 
            currentPatientWillAkd = dfWillAkd[dfWillAkd["stay_id"] == row["stay_id"]]
            isAkd = any(
                (startTime <= currentPatientWillAkd["charttime"]) &
                (currentPatientWillAkd["charttime"] < endTime) &
                (dfWillAkd["is_akd"])
            ) 
            if (isAkd):
                continue

            # get last group, check will_akd
            newDf = pd.DataFrame(data=[row], copy=True, columns= dfPatient.columns)
            willAkd = any(
                (endTime <= currentPatientWillAkd["charttime"]) &
                (currentPatientWillAkd["charttime"] < nextTime) &
                (dfWillAkd["is_akd"])
            )
            newDf["will_akd"] = willAkd
            
            # find mesures 
            for mesureName in mesuresName:
                dfCurrentMesure: pd.DataFrame = mesureMap[mesureName]
                dfCurrentMesure = dfCurrentMesure[dfCurrentMesure["stay_id"] == row["stay_id"]]
                
                mesureValues = []
                for ii in range(mesureGroupLength):
                    lowTime = startTime + ii * mesureInterval
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
            
            newDf.to_csv(targetCsvPath, mode="a", header=False, index=False)
            pass
        pass

    # mainDf = pd.concat(dfs)
    # mainDf.to_csv(TEMP_PATH / "maindf.csv")
    
    pass

if __name__ == "__main__":
    dfPatient = pd.read_csv(TEMP_PATH / "dfPatient.csv", parse_dates=["intime","outtime"])
    
    MESURE_INTERVAL = pd.Timedelta(hours=6)
    MESURE_GROUP_LENGTH = 6

    mergeData(dfPatient, MESURE_INTERVAL, MESURE_GROUP_LENGTH)
    pass