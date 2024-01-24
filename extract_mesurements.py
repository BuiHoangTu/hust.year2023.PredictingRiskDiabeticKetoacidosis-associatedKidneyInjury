from pathlib import Path
import pandas as pd

# mimic PATH (in which there are hosp and icu data)
MIMIC_PATH = Path("../mimiciv2.2/")

# temporary path
TEMP_PATH = Path("tmp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)

# other important mesurements 
IMPORTANT_MESUREMENTS = {
    51006: "bun",
    227519: "urine_output",
    224639: "weight",
    227457: "plt"
}

def extract_chartevents_mesurement(mesureId: int, outputFile: str):
    mesureChunks = []

    targetPatients = set(pd.read_csv(TEMP_PATH / "icu_target.csv", usecols=["stay_id"])["stay_id"])

    for chunk in pd.read_csv(MIMIC_PATH / "icu" / "chartevents.csv", chunksize=10000):
        isCreatinineRow = chunk["itemid"] == mesureId
        isInTargetPatients = chunk["stay_id"].isin(targetPatients)
        
        filteredChunk = chunk[isCreatinineRow & isInTargetPatients]
        mesureChunks.append(filteredChunk)
        pass
    dfMesure = pd.concat(mesureChunks)
    
    dfMesure.to_csv(TEMP_PATH / outputFile)
    pass

if __name__ == "__main__":
    for icdCode, icdName in IMPORTANT_MESUREMENTS.items():
        extract_chartevents_mesurement(icdCode, "chartevent_" + icdName + ".csv")