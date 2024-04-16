# Note 
## Note_note
- `valuenum` is `value` as float

## result of sql vs pure python 
1. number of **non-null** result are same
1. result are same 
1. SQL consist of null values 

## Require
1. Data of people who caught DKA and need to go to icu
1. Mark when they went bad 
1. collect other data (period of 3 times)
    1. Bun (blood urea nitrogen) test - labevent - 51006, 52647
    1. urine output - icu - 227519
    1. weight (daily) - icu - 224639
    1. age 
    1. plt (platelet) - icu - 227457
1. remove 9999 values 


i have 
    1. df_user containing static data: user_id, user_static_data
    2. df_usage containing which user use the app at when, and mesurements when they use the app: usage_id, user_id, mesure_time, mesure_name, mesure_value
    3. array of target_mesure_name that im interested in
i need to take each target_mesure_name, find the right user, add it as the new column named target_mesure_name_{i}, `i` in 1,2,3. Assuming each user only have 3 usage 
## most frequent mesures 
0  227969  Safety Measures
1  220045  Heart Rate
2  220210  Respiratory Rate
3  220277  O2 saturation pulseoxymetry
4  220048  Heart Rhythm
5  224650  Ectopy Type
6  220179  Non Invasive Blood Pressure systolic
7  220180  Non Invasive Blood Pressure diastolic
8  220181  Non Invasive Blood Pressure mean
9  225664  Glucose finger stick