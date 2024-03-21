import pandas as pd

# Load data from CSV
df_icu = pd.read_csv('icustays.csv')
df_lab = pd.read_csv('labevents.csv')

# Filter labevents for creatinine values and valid values
df_lab = df_lab[df_lab['ITEMID'] == 50912]
df_lab = df_lab.dropna(subset=['VALUENUM']) # TODO: check if 0 is null 

# Convert charttime to datetime
df_icu['intime'] = pd.to_datetime(df_icu['intime'])
df_icu['outtime'] = pd.to_datetime(df_icu['outtime'])
df_lab['charttime'] = pd.to_datetime(df_lab['charttime'])

# Create creat_measure DataFrame
creat_measure = pd.merge(df_icu, df_lab, how='left', left_on='subject_id', right_on='subject_id')
creat_measure = creat_measure[(creat_measure['charttime'] >= (creat_measure['intime'] - pd.Timedelta(days=7))) & (creat_measure['charttime'] <= (creat_measure['intime'] + pd.Timedelta(days=7)))]

# Function to calculate lowest value in previous n hours
def lowest_value_past_n_hours(creat_measure, n):
    return creat_measure.groupby('icustay_id').apply(lambda x: x.set_index('charttime').rolling(f'{n}H')['value'].min().shift(1)).reset_index()

# Calculate lowest values in previous 48 hours and 7 days
creat48 = lowest_value_past_n_hours(creat_measure, 48)
creat7 = lowest_value_past_n_hours(creat_measure, 7)

# Merge with creat_measure to add lowest values
creat_measure = pd.merge(creat_measure, creat48, how='left', left_on=['icustay_id', 'charttime'], right_on=['icustay_id', 'charttime'])
creat_measure = pd.merge(creat_measure, creat7, how='left', left_on=['icustay_id', 'charttime'], right_on=['icustay_id', 'charttime'])

# Group by and calculate minimum values
kdigo_creat = creat_measure.groupby(['icustay_id', 'charttime', 'value']).agg({
    'value_x': 'min',
    'value_y': 'min'
}).reset_index().rename(columns={'value_x': 'creat_low_past_48hr', 'value_y': 'creat_low_past_7day'})

# Sort the result
kdigo_creat = kdigo_creat.sort_values(by=['icustay_id', 'charttime', 'value'])


##################################################################################

import pandas as pd

# Load data from CSV
df_icu = pd.read_csv('icustays.csv')
df_creat = pd.read_csv('kdigo_creat.csv')
df_uo = pd.read_csv('kdigo_uo.csv')

# Merge with ICU stays data
df_creat = pd.merge(df_icu, df_creat, how='left', on='icustay_id')
df_uo = pd.merge(df_icu, df_uo, how='left', on='icustay_id')

# Calculate AKI stage based on creatinine
df_creat['aki_stage_creat'] = pd.cut(x=df_creat['value'],
                                     bins=[-float('inf'), 0.5 * df_creat['creat_low_past_7day'],
                                           df_creat['creat_low_past_48hr'] + 0.3, df_creat['creat_low_past_7day'] * 1.5,
                                           df_creat['creat_low_past_7day'] * 2.0, float('inf')],
                                     labels=[0, 1, 1, 2, 3])

# Calculate AKI stage based on urine output
df_uo['aki_stage_uo'] = pd.cut(df_uo['uo_rt_6hr'],
                                bins=[-float('inf'), 0.5, 0.5, 0.3, float('inf')],
                                labels=[0, 1, 2, 3])

# Get all charttimes documented
tm_stg = pd.concat([df_creat[['icustay_id', 'charttime']], df_uo[['icustay_id', 'charttime']]])

# Merge with ICU stays
df_kdigo_stages = pd.merge(df_icu, tm_stg, how='left', on='icustay_id')

# Merge with creatinine data
df_kdigo_stages = pd.merge(df_kdigo_stages, df_creat, how='left', on=['icustay_id', 'charttime'])

# Merge with urine output data
df_kdigo_stages = pd.merge(df_kdigo_stages, df_uo, how='left', on=['icustay_id', 'charttime'])

# Classify AKI using both creatinine and urine output criteria
df_kdigo_stages['aki_stage'] = df_kdigo_stages[['aki_stage_creat', 'aki_stage_uo']].max(axis=1)

# Sort the result
df_kdigo_stages = df_kdigo_stages.sort_values(by=['icustay_id', 'charttime'])

# Reset index
df_kdigo_stages.reset_index(drop=True, inplace=True)

# Save to CSV or use as needed
df_kdigo_stages.to_csv('kdigo_stages.csv', index=False)


##########################################################################################
# Load data from CSV
df_icu = pd.read_csv('icustays.csv')
df_kdigo_stages = pd.read_csv('kdigo_stages.csv')

# Filter data for the first 7 days
df_kdigo_stages_7day = df_kdigo_stages[(df_kdigo_stages['charttime_creat'] > (df_icu['intime'] - pd.Timedelta(hours=6))) &
                                        (df_kdigo_stages['charttime_creat'] <= (df_icu['intime'] + pd.Timedelta(days=7)))]

# Get the worst staging of creatinine in the first 48 hours
cr_aki = df_kdigo_stages_7day[df_kdigo_stages_7day['aki_stage_creat'].notnull()]
cr_aki = cr_aki.sort_values(by=['icustay_id', 'aki_stage_creat', 'value'], ascending=[True, False, False])
cr_aki = cr_aki.groupby('icustay_id').head(1).reset_index(drop=True)

# Get the worst staging of urine output in the first 48 hours
uo_aki = df_kdigo_stages_7day[df_kdigo_stages_7day['aki_stage_uo'].notnull()]
uo_aki = uo_aki.sort_values(by=['icustay_id', 'aki_stage_uo', 'uo_rt_24hr', 'uo_rt_12hr', 'uo_rt_6hr'], ascending=[True, False, False, False, False])
uo_aki = uo_aki.groupby('icustay_id').head(1).reset_index(drop=True)

# Merge with ICU stays
df_kdigo_stages_7day = pd.merge(df_icu, cr_aki, how='left', on='icustay_id')
df_kdigo_stages_7day = pd.merge(df_kdigo_stages_7day, uo_aki, how='left', on='icustay_id')

# Classify AKI using both creatinine and urine output criteria
df_kdigo_stages_7day['aki_stage_7day'] = df_kdigo_stages_7day[['aki_stage_creat', 'aki_stage_uo']].max(axis=1)
df_kdigo_stages_7day['aki_7day'] = df_kdigo_stages_7day['aki_stage_7day'].apply(lambda x: 1 if x > 0 else 0)

# Select relevant columns
df_kdigo_stages_7day = df_kdigo_stages_7day[['icustay_id', 'charttime_creat', 'value', 'aki_stage_creat',
                                             'charttime_uo', 'uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr',
                                             'aki_stage_uo', 'aki_stage_7day', 'aki_7day']]

# Sort the result
df_kdigo_stages_7day = df_kdigo_stages_7day.sort_values(by='icustay_id')

# Reset index
df_kdigo_stages_7day.reset_index(drop=True, inplace=True)

# Save to CSV or use as needed
df_kdigo_stages_7day.to_csv('kdigo_stages_7day.csv', index=False)
