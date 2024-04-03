WITH cr_aki AS (
  SELECT k.stay_id,
    k.charttime,
    k.creat,
    k.aki_stage_creat,
    ROW_NUMBER() OVER (
      PARTITION BY k.stay_id
      ORDER BY k.aki_stage_creat DESC,
        k.creat DESC
    ) AS rn
  FROM icustays ie
    INNER JOIN kdigo_stages k ON ie.stay_id = k.stay_id
  WHERE k.charttime > datetime(ie.intime, '-6 hours')
    AND k.charttime <= datetime(ie.intime, '+7 days')
    AND k.aki_stage_creat IS NOT NULL
) -- get the worst staging of urine output in the first 48 hours
,
uo_aki AS (
  SELECT k.stay_id,
    k.charttime,
    k.uo_rt_6hr,
    k.uo_rt_12hr,
    k.uo_rt_24hr,
    k.aki_stage_uo,
    ROW_NUMBER() OVER (
      PARTITION BY k.stay_id
      ORDER BY k.aki_stage_uo DESC,
        k.uo_rt_24hr DESC,
        k.uo_rt_12hr DESC,
        k.uo_rt_6hr DESC
    ) AS rn
  FROM icustays ie
    INNER JOIN kdigo_stages k ON ie.stay_id = k.stay_id
  WHERE k.charttime > datetime(ie.intime, '-6 hours')
    AND k.charttime <= datetime(ie.intime, '+7 days')
    AND k.aki_stage_uo IS NOT NULL
) -- final table is aki_stage, include worst cr/uo for convenience
select ie.stay_id,
  cr.charttime as charttime_creat,
  cr.creat,
  cr.aki_stage_creat,
  uo.charttime as charttime_uo,
  uo.uo_rt_6hr,
  uo.uo_rt_12hr,
  uo.uo_rt_24hr,
  uo.aki_stage_uo -- Classify AKI using both creatinine/urine output criteria
,
  CASE
    WHEN uo.aki_stage_uo IS NULL THEN cr.aki_stage_creat
    WHEN cr.aki_stage_creat >= uo.aki_stage_uo THEN cr.aki_stage_creat
    ELSE uo.aki_stage_uo
  END AS aki_stage_7day,
  CASE
    WHEN cr.aki_stage_creat > 0
    OR uo.aki_stage_uo > 0 THEN 1
    ELSE 0
  END AS aki_7day
FROM icustays ie
  LEFT JOIN cr_aki cr ON ie.stay_id = cr.stay_id
  AND cr.rn = 1
  LEFT JOIN uo_aki uo ON ie.stay_id = uo.stay_id
  AND uo.rn = 1
order by ie.stay_id;