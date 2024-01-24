from extract_mesurements import extract_chartevents_mesurement

if __name__ == "__main__":
    
    extract_chartevents_mesurement(225664, "glucose_stick.csv")
    
    extract_chartevents_mesurement(226537, "glucose_whole_blood.csv")
    
    # no sb
    # extract_chartevents_mesurement(228388, "glucose_soft_blood.csv")
    
    extract_chartevents_mesurement(220621, "glucose_serum.csv")
    