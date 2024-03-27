import pandas as pd
from pandas.io.parsers import TextFileReader

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


def extractWithStayId(
    itemId: int | list[int], source: TextFileReader, outputFile: str | None
) -> pd.DataFrame:

    if isinstance(itemId, int):
        itemId = [itemId]

    mesureChunks = []

    targetPatients = set(pd.read_csv(TEMP_PATH / "target_patients.csv", usecols=["stay_id"])["stay_id"])

    for chunk in source:
        # remove chunk id
        chunk = chunk.iloc[:, 1:]

        isIdRow = chunk["itemid"] in itemId
        isInTargetPatients = chunk["stay_id"].isin(targetPatients)

        filteredChunk = chunk[isIdRow & isInTargetPatients]
        mesureChunks.append(filteredChunk)
        pass
    dfMesure = pd.concat(mesureChunks)

    if outputFile:
        dfMesure.to_csv(TEMP_PATH / outputFile)
        pass

    return dfMesure


def extractOutputEvents(itemId: int|list[int], outputFile:str|None) -> pd.DataFrame:
    """Extract chartevent of my target patients

    Args:
        mesureId (int|list[int]): id of the mesure(s) need extracting
        outputFile (str | None): File name to store after extract

    Returns:
        pd.DataFrame: mesure and its data
    """

    source = pd.read_csv(MIMIC_PATH / "icu" / "outputevents.csv", chunksize=10000)
    
    return extractWithStayId(itemId, source, outputFile)


def extractChartEventMesures(itemId: int|list[int], outputFile: str|None) -> pd.DataFrame:
    """Extract chartevent of my target patients

    Args:
        mesureId (int|list[int]): id of the mesure(s) need extracting
        outputFile (str | None): File name to store after extract

    Returns:
        pd.DataFrame: mesure and its data
    """

    source = pd.read_csv(MIMIC_PATH / "icu" / "chartevents.csv", chunksize=10000)
    
    return extractWithStayId(itemId, source, outputFile)


def extractWithHadmId(
    itemId: int | list[int], source: TextFileReader, outputFile: str | None
) -> pd.DataFrame:
    if isinstance(itemId, int):
        itemId = [itemId]

    mesureChunks = []

    targetPatients = set(
        pd.read_csv(TEMP_PATH / "target_patients.csv", usecols=["hadm_id"])["hadm_id"]
    )

    for chunk in source:
        # remove chunk id
        chunk = chunk.iloc[:, 1:]

        isIdRow = chunk["itemid"] in itemId
        isInTargetPatients = chunk["hadm_id"].isin(targetPatients)

        filteredChunk = chunk[isIdRow & isInTargetPatients]
        mesureChunks.append(filteredChunk)
        pass
    dfMesure = pd.concat(mesureChunks)

    if outputFile:
        dfMesure.to_csv(TEMP_PATH / outputFile)
        pass

    return dfMesure


def extractLabEventMesures(mesureId: int|list[int], outputFile: str|None) -> pd.DataFrame:
    """Extract labevent of my target patients

    Args:
        mesureId (int|list[int]): id of the mesure(s) need extracting
        outputFile (str | None): File name to store after extract

    Returns:
        pd.DataFrame: mesure and its data
    """
    
    source = pd.read_csv(MIMIC_PATH / "hosp" / "labevents.csv", chunksize=10000)
    
    return extractWithHadmId(mesureId, source, outputFile)


if __name__ == "__main__":
    # extractChartEventMesures()
    
    # for icdCode, icdName in IMPORTANT_MESUREMENTS_ICU.items():
    #     extract_chartevents_mesurement_from_icu(icdCode, "chartevent_" + icdName + ".csv")
    #     pass
    # for icdCode, icdName in IMPORTANT_MESUREMENTS_LABEVENT.items():
    #     extract_chartevents_mesurement_from_labevent(icdCode, "labevent_" + icdName + ".csv")
    #     pass
    pass
