-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
/* This query extracts dose+durations of dopamine administration */ /* Local hospital dosage guidance: 2 mcg/kg/min (low) - 10 mcg/kg/min (high) */
SELECT
  stay_id,
  linkorderid, /* all rows in mcg/kg/min */
  rate AS vaso_rate,
  amount AS vaso_amount,
  starttime,
  endtime
FROM inputevents
WHERE
  itemid = 221662 /* dopamine */