SELECT ie.subject_id,
    ie.stay_id,
    FIRST_VALUE(heart_rate) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN heart_rate IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS heart_rate_first,
    FIRST_VALUE(sbp) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN sbp IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS sbp_first,
    FIRST_VALUE(dbp) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN dbp IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS dbp_first,
    FIRST_VALUE(mbp) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN mbp IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS mbp_first,
    FIRST_VALUE(resp_rate) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN resp_rate IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS resp_rate_first,
    FIRST_VALUE(temperature) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN temperature IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS temperature_first,
    FIRST_VALUE(spo2) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN spo2 IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS spo2_first,
    FIRST_VALUE(glucose) OVER (
        PARTITION BY ce.stay_id
        ORDER BY (
                CASE
                    WHEN glucose IS NOT NULL THEN 0
                    ELSE 1
                END
            ),
            charttime
    ) AS glucose_first
FROM icustays AS ie
    LEFT JOIN vitalsign AS ce ON ie.stay_id = ce.stay_id
    AND ce.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND ce.charttime <= ie.intime + INTERVAL '1 DAY'
