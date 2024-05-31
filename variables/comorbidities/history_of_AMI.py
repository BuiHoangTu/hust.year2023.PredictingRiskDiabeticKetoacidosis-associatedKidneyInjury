import pandas as pd

from constants import MIMIC_PATH
from notebook_wrappers.target_patients_wrapper import getTargetPatientIcu


def get():
    """_summary_

    Returns:
        pandas.DataFrame: ["hadm_id", "history_ami"]
    """

    amiCodePrefixes = [
        "4100",
        "4101",
        "4102",
        "4103",
        "4104",
        "4105",
        "4108",
        "4109",
        # "410",
        # "I23",
    ]

    amiFullCodes = [
        "412",  # Old
        "I21",
        "I219",
        "I230",
        "I231",
        "I232",
        "I233",
        "I234",
        "I235",
        "I236",
        "I238",
        "I252",  # Old
    ]
    dfPatIcu = getTargetPatientIcu()
    dfPatIcu = dfPatIcu[["subject_id", "hadm_id"]]
    dfPatIcu.sort_values("subject_id", inplace=True)

    patients = set(dfPatIcu["subject_id"])

    dfIcd = pd.read_csv(MIMIC_PATH / "hosp" / "diagnoses_icd.csv")
    dfIcd = dfIcd[dfIcd["subject_id"].isin(patients)]

    dfIcdPrevHadm = dfIcd.merge(dfPatIcu, on="subject_id", suffixes=("_all", "_target"))
    dfIcdPrevHadm = dfIcdPrevHadm[
        dfIcdPrevHadm["hadm_id_all"] <= dfIcdPrevHadm["hadm_id_target"]
    ]
    dfIcdPrevHadm = dfIcdPrevHadm.rename(columns={"hadm_id_all": "hadm_id"})

    dfIcuPrevAmi = dfIcdPrevHadm[
        dfIcdPrevHadm["icd_code"].isin(amiFullCodes)
        | dfIcdPrevHadm["icd_code"].str.startswith(tuple(amiCodePrefixes))
    ]

    df = pd.DataFrame(
        {
            "hadm_id": list(set(dfIcuPrevAmi["hadm_id_target"])),
            "history_ami": True,
        }
    )

    return df.sort_values("hadm_id")
