with creat_stg AS
(
  SELECT
    creat_mesure.stay_id
    , creat_mesure.charttime
    , creat_mesure.valuenum as value
    , aki_stage_creat
  FROM kdigo_creat creat_mesure
)
-- stages for UO / creat
, urine_stg as
(
  select
      urine.stay_id
    , urine.charttime
    , urine.weight
    , urine.uo_rt_6hr
    , urine.uo_rt_12hr
    , urine.uo_rt_24hr
    -- AKI stages according to urine output
    , CASE
        WHEN urine.uo_rt_6hr IS NULL THEN NULL
        -- require patient to be in ICU for at least 6 hours to stage UO
        WHEN urine.charttime <= datetime(icu.intime, '+6 hour') THEN 0
        -- require the UO rate to be calculated over half the period
        -- i.e. for urine rate over 24 hours, require documentation at least 12 hr apart
        WHEN urine.uo_tm_24hr >= 11 AND urine.uo_rt_24hr < 0.3 THEN 3
        WHEN urine.uo_tm_12hr >= 5 AND urine.uo_rt_12hr = 0 THEN 3
        WHEN urine.uo_tm_12hr >= 5 AND urine.uo_rt_12hr < 0.5 THEN 2
        WHEN urine.uo_tm_6hr >= 2 AND urine.uo_rt_6hr  < 0.5 THEN 1
    ELSE 0 END AS aki_stage_uo
  from kdigo_uo urine
  INNER JOIN icustays icu
    ON urine.stay_id = icu.stay_id
)
-- get all charttimes documented
, tm_stg AS
(
    SELECT
      stay_id, charttime
    FROM creat_stg
    UNION
    SELECT
      stay_id, charttime
    FROM urine_stg
)
select
    icu.stay_id
  , tm.charttime
  , creat_mesure.value
  , creat_mesure.aki_stage_creat
  , urine.uo_rt_6hr
  , urine.uo_rt_12hr
  , urine.uo_rt_24hr
  , urine.aki_stage_uo
  -- Classify AKI using both creat/urine output criteria
  , CASE
      WHEN creat_mesure.aki_stage_creat >= urine.aki_stage_uo THEN creat_mesure.aki_stage_creat
      ELSE urine.aki_stage_uo
    END AS aki_stage
FROM icustays icu
-- get all possible charttimes as listed in tm_stg
LEFT JOIN tm_stg tm
  ON icu.stay_id = tm.stay_id
LEFT JOIN creat_stg creat_mesure
  ON icu.stay_id = creat_mesure.stay_id
  AND tm.charttime = creat_mesure.charttime
LEFT JOIN urine_stg urine
  ON icu.stay_id = urine.stay_id
  AND tm.charttime = urine.charttime
order by icu.stay_id, tm.charttime;