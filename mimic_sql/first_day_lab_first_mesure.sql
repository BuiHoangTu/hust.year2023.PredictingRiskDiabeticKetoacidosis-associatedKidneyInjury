WITH cbc AS (
  SELECT ie.stay_id,
    FIRST_VALUE(hematocrit) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN hematocrit IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS hematocrit_first,
    FIRST_VALUE(hemoglobin) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN hemoglobin IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS hemoglobin_first,
    FIRST_VALUE(platelet) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN platelet IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS platelets_first,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN wbc IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS wbc_first
  FROM icustays AS ie
    LEFT JOIN complete_blood_count AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
),
chem AS (
  SELECT ie.stay_id,
    FIRST_VALUE(albumin) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN albumin IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS albumin_first,
    FIRST_VALUE(globulin) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN globulin IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS globulin_first,
    FIRST_VALUE(total_protein) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN total_protein IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS total_protein_first,
    FIRST_VALUE(aniongap) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN aniongap IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS aniongap_first,
    FIRST_VALUE(bicarbonate) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN bicarbonate IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS bicarbonate_first,
    FIRST_VALUE(bun) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN bun IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS bun_first,
    FIRST_VALUE(calcium) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN calcium IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS calcium_first,
    FIRST_VALUE(chloride) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN chloride IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS chloride_first,
    FIRST_VALUE(creatinine) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN creatinine IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS creatinine_first,
    FIRST_VALUE(glucose) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN glucose IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS glucose_first,
    FIRST_VALUE(sodium) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN sodium IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS sodium_first,
    FIRST_VALUE(potassium) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN potassium IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS potassium_first
  FROM icustays AS ie
    LEFT JOIN chemistry AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
),
diff AS (
  SELECT ie.stay_id,
    FIRST_VALUE(basophils_abs) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN basophils_abs IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS abs_basophils_first,
    FIRST_VALUE(eosinophils_abs) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN eosinophils_abs IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS abs_eosinophils_first,
    FIRST_VALUE(lymphocytes_abs) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN lymphocytes_abs IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS abs_lymphocytes_first,
    FIRST_VALUE(monocytes_abs) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN monocytes_abs IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS abs_monocytes_first,
    FIRST_VALUE(neutrophils_abs) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN neutrophils_abs IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS abs_neutrophils_first,
    FIRST_VALUE(atypical_lymphocytes) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN atypical_lymphocytes IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS atyps_first,
    FIRST_VALUE(bands) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN bands IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS bands_first,
    FIRST_VALUE(immature_granulocytes) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN immature_granulocytes IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS imm_granulocytes_first,
    FIRST_VALUE(metamyelocytes) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN metamyelocytes IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS metas_first,
    FIRST_VALUE(nrbc) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN nrbc IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS nrbc_first
  FROM icustays AS ie
    LEFT JOIN blood_differential AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
),
coag AS (
  SELECT ie.stay_id,
    FIRST_VALUE(d_dimer) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN d_dimer IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS d_dimer_first,
    FIRST_VALUE(fibrinogen) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN fibrinogen IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS fibrinogen_first,
    FIRST_VALUE(thrombin) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN thrombin IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS thrombin_first,
    FIRST_VALUE(inr) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN inr IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS inr_first,
    FIRST_VALUE(pt) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN pt IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS pt_first,
    FIRST_VALUE(ptt) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN ptt IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS ptt_first
  FROM icustays AS ie
    LEFT JOIN coagulation AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
),
enz AS (
  SELECT ie.stay_id,
    FIRST_VALUE(alt) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN alt IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS alt_first,
    FIRST_VALUE(alp) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN alp IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS alp_first,
    FIRST_VALUE(ast) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN ast IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS ast_first,
    FIRST_VALUE(amylase) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN amylase IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS amylase_first,
    FIRST_VALUE(bilirubin_total) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN bilirubin_total IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS bilirubin_total_first,
    FIRST_VALUE(bilirubin_direct) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN bilirubin_direct IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS bilirubin_direct_first,
    FIRST_VALUE(bilirubin_indirect) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN bilirubin_indirect IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS bilirubin_indirect_first,
    FIRST_VALUE(ck_cpk) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN ck_cpk IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS ck_cpk_first,
    FIRST_VALUE(ck_mb) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN ck_mb IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS ck_mb_first,
    FIRST_VALUE(ggt) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN ggt IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS ggt_first,
    FIRST_VALUE(ld_ldh) OVER (
      PARTITION BY stay_id
      ORDER BY (
          CASE
            WHEN ld_ldh IS NOT NULL THEN 0
            ELSE 1
          END
        ),
        charttime
    ) AS ld_ldh_first
  FROM icustays AS ie
    LEFT JOIN enzyme AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
)
SELECT ie.subject_id,
  ie.stay_id,
  /* complete blood count */
  cbc.hematocrit_first,
  cbc.hemoglobin_first,
  cbc.platelets_first,
  cbc.wbc_first,
  /* chemistry */
  chem.albumin_first,
  chem.globulin_first,
  chem.total_protein_first,
  chem.aniongap_first,
  chem.bicarbonate_first,
  chem.bun_first,
  chem.calcium_first,
  chem.chloride_first,
  chem.creatinine_first,
  chem.glucose_first,
  chem.sodium_first,
  chem.potassium_first,
  /* blood differential */
  diff.abs_basophils_first,
  diff.abs_eosinophils_first,
  diff.abs_lymphocytes_first,
  diff.abs_monocytes_first,
  diff.abs_neutrophils_first,
  diff.atyps_first,
  diff.bands_first,
  diff.imm_granulocytes_first,
  diff.metas_first,
  diff.nrbc_first,
  /* coagulation */
  coag.d_dimer_first,
  coag.fibrinogen_first,
  coag.thrombin_first,
  coag.inr_first,
  coag.pt_first,
  coag.ptt_first,
  /* enzymes and bilirubin */
  enz.alt_first,
  enz.alp_first,
  enz.ast_first,
  enz.amylase_first,
  enz.bilirubin_total_first,
  enz.bilirubin_direct_first,
  enz.bilirubin_indirect_first,
  enz.ck_cpk_first,
  enz.ck_mb_first,
  enz.ggt_first,
  enz.ld_ldh_first
FROM icustays AS ie
  LEFT JOIN cbc ON ie.stay_id = cbc.stay_id
  LEFT JOIN chem ON ie.stay_id = chem.stay_id
  LEFT JOIN diff ON ie.stay_id = diff.stay_id
  LEFT JOIN coag ON ie.stay_id = coag.stay_id
  LEFT JOIN enz ON ie.stay_id = enz.stay_id;