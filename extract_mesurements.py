import pandas as pd

from constants import MIMIC_PATH, TEMP_PATH

# other important mesurements 
IMPORTANT_MESUREMENTS_ICU = {
    227519: "urine_output",
    224639: "weight",
    227457: "plt",
    220615: "creatinine"
}

IMPORTANT_MESUREMENTS_LABEVENT = {
    51006: "bun"    
}

def extract_chartevents_mesurement_from_icu(mesureId: int, outputFile: str|None) -> pd.DataFrame:
    mesureChunks = []

    targetPatients = set(pd.read_csv(TEMP_PATH / "icu_target.csv", usecols=["stay_id"])["stay_id"])

    for chunk in pd.read_csv(MIMIC_PATH / "icu" / "chartevents.csv", chunksize=10000):
        # remove chunk id 
        chunk = chunk.iloc[:, 1:]
        
        isCreatinineRow = chunk["itemid"] == mesureId
        isInTargetPatients = chunk["stay_id"].isin(targetPatients)
        
        filteredChunk = chunk[isCreatinineRow & isInTargetPatients]
        mesureChunks.append(filteredChunk)
        pass
    dfMesure = pd.concat(mesureChunks)
    
    if (outputFile):
        dfMesure.to_csv(TEMP_PATH / outputFile)
        pass
    
    return dfMesure


def extract_chartevents_mesurement_from_labevent(mesureId: int, outputFile: str|None) -> pd.DataFrame:
    mesureChunks = []

    targetPatients = set(pd.read_csv(TEMP_PATH / "icu_target.csv", usecols=["hadm_id"])["hadm_id"])

    for chunk in pd.read_csv(MIMIC_PATH / "hosp" / "labevents.csv", chunksize=10000):
        # remove chunk id 
        chunk = chunk.iloc[:, 1:]
        
        isCreatinineRow = chunk["itemid"] == mesureId
        isInTargetPatients = chunk["hadm_id"].isin(targetPatients)
        
        filteredChunk = chunk[isCreatinineRow & isInTargetPatients]
        mesureChunks.append(filteredChunk)
        pass
    dfMesure = pd.concat(mesureChunks)
    
    if (outputFile):
        dfMesure.to_csv(TEMP_PATH / outputFile)
        pass
    
    return dfMesure

if __name__ == "__main__":
    for icdCode, icdName in IMPORTANT_MESUREMENTS_ICU.items():
        extract_chartevents_mesurement_from_icu(icdCode, "chartevent_" + icdName + ".csv")
        pass
    for icdCode, icdName in IMPORTANT_MESUREMENTS_LABEVENT.items():
        extract_chartevents_mesurement_from_labevent(icdCode, "labevent_" + icdName + ".csv")
        pass
    pass
