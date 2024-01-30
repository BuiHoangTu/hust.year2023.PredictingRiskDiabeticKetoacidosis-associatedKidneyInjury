import pandas as pd
import numpy as np
from extract_mesurements import extract_chartevents_mesurement_from_icu


CREATININE_ID = 220615
CREATININE_FRAC = 1.5
CREATININE_DELTA = 0.3
CREATININE_MAX = 15/133.12 * 10  ## convert 15mg/mmol to mg/dl 

def markWillAkd():
    groups = markIsAkd()
    for groupId, groupDf in groups :
        groupDfSorted = groupDf.sort_values(by="charttime")
        isAkdIndexes = groupDfSorted[groupDfSorted["is_akd"]].index
        
        if (len(isAkdIndexes) > 0):
            lastTimeIsAkd: np.DateTime
            for index, rowDf in groupDfSorted.iterrows():
                if (index < isAkdIndexes[0]): continue
                if (\
                    (not lastTimeIsAkd) or\
                    (rowDf["is_akd"] and (rowDf["charttime"] < lastTimeIsAkd + np.timedelta64(6, "h")))\
                ): lastTimeIsAkd = rowDf["charttime"]
                if (rowDf["charttime"])
                pass 
            # Remove later mesure if already caugh akd once 
            firstIdx = isAkdIndexes[0]
            x = groupDfSorted.loc[firstIdx]
            # groupNoAkd: pd.DataFrame = groupDfSorted.drop(index=range(firstIdx, len(groupDfSorted)))
            
            # if first index is at top, dont mark will akd 
            # if (firstIdx == groupDfSorted.index[0]):
            #     isAkdIndexes = isAkdIndexes[1:]
            #     pass
            
            groupDfSorted["will_akd"] = groupDfSorted["is_akd"].shift(-1, fill_value=False)
            pass
        
        groupDf
        pass

    pass

def markIsAkd():
    extract_chartevents_mesurement_from_icu(CREATININE_ID, "chartevent_creatinine.csv")
    dfCreatinine = pd.read_csv("/tmp/chartevent_creatinine.csv")
    
    # mark by max value 
    dfCreatinine.loc[:, "is_akd"] = dfCreatinine["valuenum"] >= CREATININE_MAX

    # mark by delta and max:min 
    df_icu_group = dfCreatinine.groupby("stay_id")
    for groupId, groupDf in df_icu_group :
        # mark consecutive delta 
        # make sure sorted 
        groupDf = groupDf.sort_values(by="charttime")
        groupDf["is_akd"] = groupDf["is_akd"] | (abs(groupDf['valuenum'].diff()) > CREATININE_DELTA)
    
        # mark by max/min 
        maxIdx = groupDf["valuenum"].idxmax()
        minIdx = groupDf["valuenum"].idxmin()
        
        if ((groupDf["valuenum"][minIdx] == 0) or (groupDf["valuenum"][maxIdx] / groupDf["valuenum"][minIdx] > CREATININE_FRAC)) :
            laterIdx = maxIdx if maxIdx > minIdx else minIdx  # type: ignore
            groupDf.loc[laterIdx, "is_akd"] = True
            pass
        pass
    return df_icu_group

