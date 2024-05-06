from pandas import DataFrame


def calculate_eGFR(standardized_scr: float, age: int, gender: str) -> float:
    """
    Calculate the estimated Glomerular Filtration Rate (eGFR) using the following formula:

    eGFR = 142 * min(standardized_scr/K, 1)^alpha * max(standardized_scr/K, 1)^-1.209 * 0.993^age

    If the patient is female, the result is multiplied by 1.018.

    Parameters:
    standardized_scr (float): The standardized serum creatinine level.
    age (int): The age of the patient.
    gender (str): The gender of the patient ('F' or 'M').

    Returns:
    float: The estimated Glomerular Filtration Rate (eGFR).
    """

    # Constants for the formula
    K = 0.9 if gender == "M" else 0.7
    alpha = -0.411 if gender == "M" else -0.329

    min_value = min(standardized_scr / K, 1)
    max_value = max(standardized_scr / K, 1)

    eGFR = 142 * (min_value**alpha) * (max_value**-1.209) * (0.993**age)

    # Adjust for gender
    if not gender == "M":
        eGFR *= 1.018

    return eGFR

def calculate_eGFR_df(df: DataFrame):
    """Calculate eGFR for the whole dataframe.

    Args:
        df (DataFrame): must contains scr, age, gender(str)

    Returns:
        DataFrame: contains stay_id, egfr
    """

    df["egfr"] = df.apply(
        lambda row: calculate_eGFR(row["scr"], row["age"], row["gender"] == "male"),
        axis=1,
    )

    return df[["stay_id", "egfr"]]
