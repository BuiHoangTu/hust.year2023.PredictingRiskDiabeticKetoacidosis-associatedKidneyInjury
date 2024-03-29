-- Supplementary Materials
-- ************************************CODE************************************

DROP MATERIALIZED VIEW IF EXISTS kdigo_creat CASCADE;
CREATE MATERIALIZED VIEW kdigo_creat as
-- Extract all creatinine values from labevents around patient's ICU stay
with creat_mesure as
(
select
    icu.icustay_id
  , icu.intime, icu.outtime
  , lab.valuenum as value
  , lab.charttime
  from icustays icu
  left join labevents lab
    on icu.subject_id = lab.subject_id
    and lab.ITEMID = 50912 -- creatini id
    and lab.VALUENUM is not null
    and lab.CHARTTIME between (icu.intime - interval '7' day) and (icu.intime + interval '7' day)
)

-- add in the lowest value in the previous 48 hours/7 days
SELECT
  creat_mesure.icustay_id
  , creat_mesure.charttime
  , creat_mesure.value
  , MIN(creat48.value) AS creat_low_past_48hr
  , MIN(creat7.value) AS creat_low_past_7day
FROM creat_mesure
-- add in all creatinine values in the last 48 hours
LEFT JOIN creat_mesure creat48
  ON creat_mesure.icustay_id = creat48.icustay_id
  AND creat48.charttime <  creat_mesure.charttime
  AND creat48.charttime >= (creat_mesure.charttime - INTERVAL '48' HOUR)
-- add in all creatinine values in the last 7 days hours
LEFT JOIN creat_mesure creat7
  ON creat_mesure.icustay_id = creat7.icustay_id
  AND creat7.charttime <  creat_mesure.charttime
  AND creat7.charttime >= (creat_mesure.charttime - INTERVAL '7' DAY)
GROUP BY creat_mesure.icustay_id, creat_mesure.charttime, creat_mesure.value
ORDER BY creat_mesure.icustay_id, creat_mesure.charttime, creat_mesure.value;

-- This query checks if the patient had AKI according to KDIGO.
-- AKI is calculated every time a creatinine or urine output measurement occurs.
-- Baseline creatinine is defined as the lowest creatinine in the past 7 days.

DROP MATERIALIZED VIEW IF EXISTS kdigo_stages CASCADE;
CREATE MATERIALIZED VIEW kdigo_stages AS
-- get creatinine stages
with creat_stg AS
(
  SELECT
    creat_mesure.icustay_id
    , creat_mesure.charttime
    , creat_mesure.value
    , case
        -- 3x baseline
        when creat_mesure.value >= (creat_mesure.creat_low_past_7day*3.0) then 3
        -- *OR* creat_mesure >= 4.0 with associated increase
        when creat_mesure.value >= 4
        -- For patients reaching Stage 3 by SCr >4.0 mg/dl
        -- require that the patient first achieve ... acute increase >= 0.3 within 48 hr
        -- *or* an increase of >= 1.5 times baseline
        and (creat_mesure.creat_low_past_48hr <= 3.7 OR creat_mesure.value >= (1.5*creat_mesure.creat_low_past_7day))
            then 3 
        -- TODO: initiation of RRT
        when creat_mesure.value >= (creat_mesure.creat_low_past_7day*2.0) then 2
        when creat_mesure.value >= (creat_mesure.creat_low_past_48hr+0.3) then 1
        when creat_mesure.value >= (creat_mesure.creat_low_past_7day*1.5) then 1
    else 0 end as aki_stage_creat
  FROM kdigo_creat creat_mesure
)
-- stages for UO / creat
, urine_stg as
(
  select
      urine.icustay_id
    , urine.charttime
    , urine.weight
    , urine.uo_rt_6hr
    , urine.uo_rt_12hr
    , urine.uo_rt_24hr
    -- AKI stages according to urine output
    , CASE
        WHEN urine.uo_rt_6hr IS NULL THEN NULL
        -- require patient to be in ICU for at least 6 hours to stage UO
        WHEN urine.charttime <= icu.intime + interval '6' hour THEN 0
        -- require the UO rate to be calculated over half the period
        -- i.e. for urine rate over 24 hours, require documentation at least 12 hr apart
        WHEN urine.uo_tm_24hr >= 11 AND urine.uo_rt_24hr < 0.3 THEN 3
        WHEN urine.uo_tm_12hr >= 5 AND urine.uo_rt_12hr = 0 THEN 3
        WHEN urine.uo_tm_12hr >= 5 AND urine.uo_rt_12hr < 0.5 THEN 2
        WHEN urine.uo_tm_6hr >= 2 AND urine.uo_rt_6hr  < 0.5 THEN 1
    ELSE 0 END AS aki_stage_uo
  from kdigo_uo urine
  INNER JOIN icustays icu
    ON urine.icustay_id = icu.icustay_id
)
-- get all charttimes documented
, tm_stg AS
(
    SELECT
      icustay_id, charttime
    FROM creat_stg
    UNION
    SELECT
      icustay_id, charttime
    FROM urine_stg
)
select
    icu.icustay_id
  , tm.charttime
  , creat_mesure.value
  , creat_mesure.aki_stage_creat
  , urine.uo_rt_6hr
  , urine.uo_rt_12hr
  , urine.uo_rt_24hr
  , urine.aki_stage_uo
  -- Classify AKI using both creat/urine output criteria
  , GREATEST(creat_mesure.aki_stage_creat, urine.aki_stage_uo) AS aki_stage
FROM icustays icu
-- get all possible charttimes as listed in tm_stg
LEFT JOIN tm_stg tm
  ON icu.icustay_id = tm.icustay_id
LEFT JOIN creat_stg creat_mesure
  ON icu.icustay_id = creat_mesure.icustay_id
  AND tm.charttime = creat_mesure.charttime
LEFT JOIN urine_stg urine
  ON icu.icustay_id = urine.icustay_id
  AND tm.charttime = urine.charttime
order by icu.icustay_id, tm.charttime;

-- This query checks if the patient had AKI during the first 7 days of their ICU
-- stay according to the KDIGO guideline.
-- https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf

DROP MATERIALIZED VIEW IF EXISTS kdigo_stages_7day;
CREATE MATERIALIZED VIEW kdigo_stages_7day AS
-- get the worst staging of creat in the first 48 hours
WITH cr_aki AS
(
  SELECT
    k.icustay_id
    , k.charttime
    , k.value
    , k.aki_stage_creat
    , ROW_NUMBER() OVER (PARTITION BY k.icustay_id ORDER BY k.aki_stage_creat DESC, k.value DESC) AS rn
  FROM icustays icu
  INNER JOIN kdigo_stages k
    ON icu.icustay_id = k.icustay_id
  WHERE k.charttime > (icu.intime - interval '6' hour)
  AND k.charttime <= (icu.intime + interval '7' day)
  AND k.aki_stage_creat IS NOT NULL
)
-- get the worst staging of urine output in the first 48 hours
, uo_aki AS
(
  SELECT
    k.icustay_id
    , k.charttime
    , k.uo_rt_6hr, k.uo_rt_12hr, k.uo_rt_24hr
    , k.aki_stage_uo
    , ROW_NUMBER() OVER 
    (
      PARTITION BY k.icustay_id
      ORDER BY k.aki_stage_uo DESC, k.uo_rt_24hr DESC, k.uo_rt_12hr DESC, k.uo_rt_6hr DESC
    ) AS rn
  FROM icustays icu
  INNER JOIN kdigo_stages k
    ON icu.icustay_id = k.icustay_id
  WHERE k.charttime > (icu.intime - interval '6' hour)
  AND k.charttime <= (icu.intime + interval '7' day)
  AND k.aki_stage_uo IS NOT NULL
)
-- final table is aki_stage, include worst creat_mesure/urine for convenience
select
    icu.icustay_id
  , creat_mesure.charttime as charttime_creat
  , creat_mesure.value
  , creat_mesure.aki_stage_creat
  , urine.charttime as charttime_uo
  , urine.uo_rt_6hr
  , urine.uo_rt_12hr
  , urine.uo_rt_24hr
  , urine.aki_stage_uo

  -- Classify AKI using both creat/urine output criteria
  , GREATEST(creat_mesure.aki_stage_creat,urine.aki_stage_uo) AS aki_stage_7day
  , CASE WHEN GREATEST(creat_mesure.aki_stage_creat, urine.aki_stage_uo) > 0 THEN 1 ELSE 0 END AS aki_7day

FROM icustays icu
LEFT JOIN cr_aki creat_mesure
  ON icu.icustay_id = creat_mesure.icustay_id
  AND creat_mesure.rn = 1
LEFT JOIN uo_aki urine
  ON icu.icustay_id = urine.icustay_id
  AND urine.rn = 1
order by icu.icustay_id;
*******************************************************************************