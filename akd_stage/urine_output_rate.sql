-- attempt to calculate urine output per hour
-- rate/hour is the interpretable measure of kidney function
-- though it is difficult to estimate from aperiodic point measures
-- first we get the earliest heart rate documented for the stay
WITH tm AS (
    SELECT ie.stay_id
           , MIN(charttime) AS intime_hr
           , MAX(charttime) AS outtime_hr
    FROM icustays ie
    INNER JOIN chartevents ce
        ON ie.stay_id = ce.stay_id
            AND ce.itemid = 220045
            AND ce.charttime > datetime(ie.intime, '-1 month')
            AND ce.charttime < datetime(ie.outtime, '+1 month')
    GROUP BY ie.stay_id
)

-- now calculate time since last UO measurement
, uo_tm AS (
    SELECT 
        tm.stay_id,
        CASE
            WHEN LAG(charttime) OVER w IS NULL 
                THEN CAST((strftime('%s', charttime) - strftime('%s', intime_hr)) / 60 AS INTEGER)
            ELSE CAST((strftime('%s', charttime) - strftime('%s', LAG(charttime) OVER w)) / 60 AS INTEGER)
        END AS tm_since_last_uo,
        uo.charttime,
        uo.urineoutput
    FROM tm
    INNER JOIN urine_output uo
        ON tm.stay_id = uo.stay_id
    WINDOW w as (PARTITION BY tm.stay_id ORDER BY charttime)
)

, ur_stg AS (
    SELECT io.stay_id, io.charttime,
        -- Total UO over the last 24 hours
        SUM(DISTINCT io.urineoutput) AS uo,
        -- UO over a 6 hour period
        SUM(CASE WHEN strftime('%s', io.charttime) - strftime('%s', iosum.charttime) < 21600 
            THEN iosum.urineoutput 
            ELSE null END) AS urineoutput_6hr,
        SUM(CASE WHEN strftime('%s', io.charttime) - strftime('%s', iosum.charttime) < 21600 
            THEN iosum.tm_since_last_uo 
            ELSE null END) / 60.0 AS uo_tm_6hr,
        -- UO over a 12 hour period
        SUM(CASE WHEN strftime('%s', io.charttime) - strftime('%s', iosum.charttime) < 43200 
            THEN iosum.urineoutput 
            ELSE null END) AS urineoutput_12hr,
        SUM(CASE WHEN strftime('%s', io.charttime) - strftime('%s', iosum.charttime) < 43200 
            THEN iosum.tm_since_last_uo 
            ELSE null END) / 60.0 AS uo_tm_12hr,
        -- UO over a 24 hour period
        SUM(iosum.urineoutput) AS urineoutput_24hr,
        SUM(iosum.tm_since_last_uo) / 60.0 AS uo_tm_24hr
    FROM uo_tm io
    -- Joining to get all UO measurements over a 24 hour period
    LEFT JOIN uo_tm iosum
        ON io.stay_id = iosum.stay_id
            AND io.charttime >= iosum.charttime
            AND io.charttime <= datetime(iosum.charttime, '+1 day', '-1 second')
    GROUP BY io.stay_id, io.charttime
)

SELECT
    ur.stay_id
    , ur.charttime
    , wd.weight
    , ur.uo
    , ur.urineoutput_6hr
    , ur.urineoutput_12hr
    , ur.urineoutput_24hr 
    , CASE
        WHEN
            uo_tm_6hr >= 6 THEN ROUND(
                CAST((ur.urineoutput_6hr / wd.weight / uo_tm_6hr) AS NUMERIC), 4
            )
    END AS uo_mlkghr_6hr
    , CASE
        WHEN
            uo_tm_12hr >= 12 THEN ROUND(
                CAST((ur.urineoutput_12hr / wd.weight / uo_tm_12hr) AS NUMERIC)
                , 4
            )
    END AS uo_mlkghr_12hr
    , CASE
        WHEN
            uo_tm_24hr >= 24 THEN ROUND(
                CAST((ur.urineoutput_24hr / wd.weight / uo_tm_24hr) AS NUMERIC)
                , 4
            )
    END AS uo_mlkghr_24hr
    -- time of earliest UO measurement that was used to calculate the rate
    , ROUND(CAST(uo_tm_6hr AS NUMERIC), 2) AS uo_tm_6hr
    , ROUND(CAST(uo_tm_12hr AS NUMERIC), 2) AS uo_tm_12hr
    , ROUND(CAST(uo_tm_24hr AS NUMERIC), 2) AS uo_tm_24hr
FROM ur_stg ur
LEFT JOIN weight_durations wd
    ON ur.stay_id = wd.stay_id
        AND ur.charttime > wd.starttime
        AND ur.charttime <= wd.endtime
        AND wd.weight > 0
;