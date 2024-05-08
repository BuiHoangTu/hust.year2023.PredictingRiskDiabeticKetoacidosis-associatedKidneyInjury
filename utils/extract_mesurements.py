import pandas as pd
from pandas.io.parsers import TextFileReader

from constants import MIMIC_PATH, TARGET_PATIENT_FILE, TEMP_PATH

# other important mesurements
IMPORTANT_MESUREMENTS_ICU = {
    227519: "urine_output",
    224639: "weight",
    227457: "plt",
    220615: "creatinine",
}

IMPORTANT_MESUREMENTS_LABEVENT = {
    51006: "bun",
}


def extractWithStayId(
    itemId: int | list[int], source: TextFileReader, outputFile: str | None
) -> pd.DataFrame:

    if isinstance(itemId, int):
        itemId = [itemId]

    mesureChunks = []

    targetPatients = set(
        pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE, usecols=["stay_id"])["stay_id"]
    )

    for chunk in source:
        isIdRow = chunk["itemid"].isin(itemId)
        isInTargetPatients = chunk["stay_id"].isin(targetPatients)

        filteredChunk = chunk[isIdRow & isInTargetPatients]
        mesureChunks.append(filteredChunk)
        pass
    dfMesure = pd.concat(mesureChunks)

    if outputFile:
        dfMesure.to_csv(TEMP_PATH / outputFile)
        pass

    return dfMesure


def extractOutputEvents(
    itemId: int | list[int], outputFile: str | None
) -> pd.DataFrame:
    """Extract output event of my target patients, save to outputFile if not None.
    This will try return content of outputFile beforehand.

    Args:
        mesureId (int|list[int]): id of the mesure(s) need extracting
        outputFile (str | None): File name to store after extract

    Returns:
        pd.DataFrame: mesure and its data
    """

    if outputFile is not None and (TEMP_PATH / outputFile).exists():
        res = pd.read_csv(TEMP_PATH / outputFile, parse_dates=["charttime"])

    else:
        source = pd.read_csv(
            MIMIC_PATH / "icu" / "outputevents.csv",
            chunksize=10000,
            parse_dates=["charttime"],
        )

        res = extractWithStayId(itemId, source, outputFile)

        pass

    res["charttime"] = pd.to_datetime(res["charttime"])

    return res


def extractInputEvents(itemId: int | list[int], outputFile: str | None) -> pd.DataFrame:
    """Extract input event of my target patients, save to outputFile if not None.
    This will try return content of outputFile beforehand.

    Args:
        mesureId (int|list[int]): id of the mesure(s) need extracting
        outputFile (str | None): File name to store after extract

    Returns:
        pd.DataFrame: mesure and its data
    """

    if outputFile is not None and (TEMP_PATH / outputFile).exists():
        res = pd.read_csv(TEMP_PATH / outputFile, parse_dates=["starttime", "endtime"])

    else:
        source = pd.read_csv(
            MIMIC_PATH / "icu" / "inputevents.csv",
            chunksize=10000,
            parse_dates=["starttime", "endtime"],
        )

        res = extractWithStayId(itemId, source, outputFile)

        pass

    res["starttime"] = pd.to_datetime(res["starttime"])
    res["endtime"] = pd.to_datetime(res["endtime"])

    return res


def extractChartEventMesures(
    itemId: int | list[int], outputFile: str | None
) -> pd.DataFrame:
    """Extract chartevent of my target patients, save to outputFile if not None.
    This will try return content of outputFile beforehand.

    Args:
        mesureId (int|list[int]): id of the mesure(s) need extracting
        outputFile (str | None): File name to store after extract

    Returns:
        pd.DataFrame: mesure and its data
    """

    if outputFile is not None and (TEMP_PATH / outputFile).exists():
        res = pd.read_csv(TEMP_PATH / outputFile, parse_dates=["charttime"])

    else:
        source = pd.read_csv(
            MIMIC_PATH / "icu" / "chartevents.csv",
            chunksize=10000,
            parse_dates=["charttime"],
        )

        res = extractWithStayId(itemId, source, outputFile)

        pass

    res["charttime"] = pd.to_datetime(res["charttime"])

    return res


def extractWithHadmId(
    itemId: int | list[int], source: TextFileReader, outputFile: str | None
) -> pd.DataFrame:
    if isinstance(itemId, int):
        itemId = [itemId]

    mesureChunks = []

    targetPatients = set(
        pd.read_csv(TEMP_PATH / TARGET_PATIENT_FILE, usecols=["hadm_id"])["hadm_id"]
    )

    for chunk in source:
        isIdRow = chunk["itemid"].isin(itemId)
        isInTargetPatients = chunk["hadm_id"].isin(targetPatients)

        filteredChunk = chunk[isIdRow & isInTargetPatients]
        mesureChunks.append(filteredChunk)
        pass
    dfMesure = pd.concat(mesureChunks)

    if outputFile:
        dfMesure.to_csv(TEMP_PATH / outputFile)
        pass

    return dfMesure


def extractLabEventMesures(
    mesureId: int | list[int], outputFile: str | None
) -> pd.DataFrame:
    """Extract labevent of my target patients, save to outputFile if not None.
    This will try return content of outputFile beforehand.

    Args:
        mesureId (int|list[int]): id of the mesure(s) need extracting
        outputFile (str | None): File name to store after extract

    Returns:
        pd.DataFrame: mesure and its data
    """

    if outputFile is not None and (TEMP_PATH / outputFile).exists():
        res = pd.read_csv(TEMP_PATH / outputFile, parse_dates=["charttime"])
        pass

    else:
        source = pd.read_csv(
            MIMIC_PATH / "hosp" / "labevents.csv",
            chunksize=10000,
            parse_dates=["charttime"],
        )
        res = extractWithHadmId(mesureId, source, outputFile)

        pass

    res["charttime"] = pd.to_datetime(res["charttime"])

    return res


if __name__ == "__main__":
    # extractChartEventMesures()

    # for icdCode, icdName in IMPORTANT_MESUREMENTS_ICU.items():
    #     extract_chartevents_mesurement_from_icu(icdCode, "chartevent_" + icdName + ".csv")
    #     pass
    # for icdCode, icdName in IMPORTANT_MESUREMENTS_LABEVENT.items():
    #     extract_chartevents_mesurement_from_labevent(icdCode, "labevent_" + icdName + ".csv")
    #     pass
    pass
