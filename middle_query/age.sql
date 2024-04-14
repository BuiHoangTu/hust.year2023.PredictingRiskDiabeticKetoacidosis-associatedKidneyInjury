/* This query calculates the age of a patient on admission to the hospital. */
/* The columns of the table patients: anchor_age, anchor_year, anchor_year_group */
/* provide information regarding the actual patient year for the patient */
/* admission, and the patient's age at that time. */
/* anchor_year is a shifted year for the patient. */
/* anchor_year_group is a range of years - the patient's anchor_year occurred */
/* during this range. */
/* anchor_age is the patient's age in the anchor_year. */
/* Example: a patient has an anchor_year of 2153, */
/* anchor_year_group of 2008 - 2010, and an anchor_age of 60. */
/* The year 2153 for the patient corresponds to 2008, 2009, or 2010. */
/* The patient was 60 in the shifted year of 2153, */
/*  i.e. they were 60 in 2008, 2009, or 2010. */
/* A patient admission in 2154 will occur in 2009-2011, */
/* an admission in 2155 will occur in 2010-2012, and so on. */
/* Therefore, the age of a patient = admission time - anchor_year + anchor_age */
SELECT ad.subject_id,
  ad.hadm_id,
  ad.admittime,
  pa.anchor_age,
  pa.anchor_year,
  /* calculate the age as anchor_age (60) plus difference between */
  /* admit year and the anchor year. */
  /* the noqa retains the extra long line so the */
  /* convert to postgres bash script works */
  pa.anchor_age + EXTRACT(
    EPOCH
    FROM ad.admittime - TO_TIMESTAMP(
        TO_CHAR(pa.anchor_year, '0000') || TO_CHAR(1, '00') || TO_CHAR(1, '00') || TO_CHAR(0, '00') || TO_CHAR(0, '00') || TO_CHAR(0, '00'),
        'yyyymmddHH24MISS'
      )
  ) / 31556908.8 AS age
  /* noqa: L016 */
FROM admissions AS ad
  INNER JOIN patients AS pa ON ad.subject_id = pa.subject_id