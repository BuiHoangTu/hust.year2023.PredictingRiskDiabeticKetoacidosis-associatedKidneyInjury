WITH ce AS (
  SELECT ce.subject_id,
    ce.stay_id,
    ce.charttime,
    itemid,
    /* TODO: clean */
    value,
    CASE
      WHEN itemid = 223835 THEN CASE
        WHEN valuenum >= 0.20
        AND valuenum <= 1 THEN valuenum * 100
        WHEN valuenum > 1
        AND valuenum < 20 THEN NULL
        WHEN valuenum >= 20
        AND valuenum <= 100 THEN valuenum
        ELSE NULL
      END
      WHEN itemid IN (220339, 224700) THEN CASE
        WHEN valuenum > 100 THEN NULL
        WHEN valuenum < 0 THEN NULL
        ELSE valuenum
      END
      ELSE valuenum
    END AS valuenum,
    valueuom,
    storetime
  FROM chartevents AS ce
  WHERE NOT ce.value IS NULL
    AND NOT ce.stay_id IS NULL
    AND ce.itemid IN (
      224688
      /* Respiratory Rate (Set) */
,
      224689
      /* Respiratory Rate (spontaneous) */
,
      224690
      /* Respiratory Rate (Total) */
,
      224687
      /* minute volume */
,
      224685,
      224684,
      224686
      /* tidal volume */
,
      224696
      /* PlateauPressure */
,
      220339,
      224700
      /* PEEP */
,
      223835
      /* fio2 */
,
      223849
      /* vent mode */
,
      229314
      /* vent mode (Hamilton) */
,
      223848
      /* vent type */
,
      224691
      /* Flow Rate (L) */
    )
)
SELECT subject_id,
  MAX(stay_id) AS stay_id,
  charttime,
  MAX(
    CASE
      WHEN itemid = 224688 THEN valuenum
      ELSE NULL
    END
  ) AS respiratory_rate_set,
  MAX(
    CASE
      WHEN itemid = 224690 THEN valuenum
      ELSE NULL
    END
  ) AS respiratory_rate_total,
  MAX(
    CASE
      WHEN itemid = 224689 THEN valuenum
      ELSE NULL
    END
  ) AS respiratory_rate_spontaneous,
  MAX(
    CASE
      WHEN itemid = 224687 THEN valuenum
      ELSE NULL
    END
  ) AS minute_volume,
  MAX(
    CASE
      WHEN itemid = 224684 THEN valuenum
      ELSE NULL
    END
  ) AS tidal_volume_set,
  MAX(
    CASE
      WHEN itemid = 224685 THEN valuenum
      ELSE NULL
    END
  ) AS tidal_volume_observed,
  MAX(
    CASE
      WHEN itemid = 224686 THEN valuenum
      ELSE NULL
    END
  ) AS tidal_volume_spontaneous,
  MAX(
    CASE
      WHEN itemid = 224696 THEN valuenum
      ELSE NULL
    END
  ) AS plateau_pressure,
  MAX(
    CASE
      WHEN itemid IN (220339, 224700) THEN valuenum
      ELSE NULL
    END
  ) AS peep,
  MAX(
    CASE
      WHEN itemid = 223835 THEN valuenum
      ELSE NULL
    END
  ) AS fio2,
  MAX(
    CASE
      WHEN itemid = 224691 THEN valuenum
      ELSE NULL
    END
  ) AS flow_rate,
  MAX(
    CASE
      WHEN itemid = 223849 THEN value
      ELSE NULL
    END
  ) AS ventilator_mode,
  MAX(
    CASE
      WHEN itemid = 229314 THEN value
      ELSE NULL
    END
  ) AS ventilator_mode_hamilton,
  MAX(
    CASE
      WHEN itemid = 223848 THEN value
      ELSE NULL
    END
  ) AS ventilator_type
FROM ce
GROUP BY subject_id,
  charttime;