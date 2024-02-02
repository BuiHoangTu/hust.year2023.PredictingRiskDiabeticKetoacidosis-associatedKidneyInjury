import os
from time import sleep
import pandas as pd
from extract_mesurements import extract_chartevents_mesurement_from_icu


CREATININE_ID = 220615
CREATININE_FRAC = 1.5
CREATININE_DELTA = 0.3
CREATININE_MAX = 15/133.12 * 10  ## convert 15mg/mmol to mg/dl 

def markWillAkd(mesureGroupTime: pd.Timedelta):
    groups = markIsAkd()
    dfsWillAkd = []
    for groupId, groupDf in groups :
        groupDfSorted = groupDf.sort_values(by="charttime")
        # isAkdIndexes = groupDfSorted[groupDfSorted["is_akd"]].index
        
        # if (len(isAkdIndexes) > 0):
        #     # first time is_akd
        #     lastTimeIsAkd: pd.Timestamp = groupDfSorted.loc[isAkdIndexes[0]]["charttime"]
            
        #     for index, rowDf in groupDfSorted.iterrows():
        #         # move to first is_akd
        #         if (index <= isAkdIndexes[0]): 
        #             continue
                
        #         # if is_akd again too close, relocate lastTimeIsAkd and remove elements between 
        #         mesuresBetween = []
        #         if (rowDf["charttime"] < lastTimeIsAkd + mesureGroupTime ):
        #             if (rowDf["is_akd"]):
        #                 # dont need to remove this row, all is_akd will be deleted
        #                 lastTimeIsAkd = rowDf["charttime"]
        #                 groupDfSorted.drop(mesuresBetween)
        #                 mesuresBetween.clear()
        #                 continue
        #             else:
        #                 mesuresBetween.append(rowDf)
        #                 continue
                
        #         # is akd but long after     
        #         if (rowDf["is_akd"]):
        #             lastTimeIsAkd = rowDf["charttime"]
        #             mesuresBetween.clear
        #             pass
        #         pass 
        #     pass
        groupDfSorted["will_akd"] = groupDfSorted["is_akd"].shift(-1, fill_value=False)
        
        # # drop all is currently akd 
        groupDfWillAkd = groupDfSorted
        dfsWillAkd.append(groupDfWillAkd)
        pass
    dfWillAkd = pd.concat(dfsWillAkd)

    return dfWillAkd

def markIsAkd():
    if (not os.path.exists("./tmp/chartevent_creatinine.csv")):
        extract_chartevents_mesurement_from_icu(CREATININE_ID, "chartevent_creatinine.csv")
    
    sleep(2)
    dfCreatinine = pd.read_csv("./tmp/chartevent_creatinine.csv", parse_dates=["charttime"])
    
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

if __name__ == "__main__":
    df = markWillAkd(5 * pd.Timedelta(hours=6))
    df.to_csv("./tmp/will_akd.csv")
    pass