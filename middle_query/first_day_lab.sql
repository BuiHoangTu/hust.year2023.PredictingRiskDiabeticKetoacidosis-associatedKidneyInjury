WITH cbc AS (
  SELECT ie.stay_id,
    MIN(hematocrit) AS hematocrit_min,
    MAX(hematocrit) AS hematocrit_max,
    FIRST_VALUE(hematocrit) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS hematocrit_first,
    MIN(hemoglobin) AS hemoglobin_min,
    MAX(hemoglobin) AS hemoglobin_max,
    FIRST_VALUE(hemoglobin) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS hemoglobin_first,
    MIN(platelet) AS platelets_min,
    MAX(platelet) AS platelets_max,
    FIRST_VALUE(platelet) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS platelets_first,
    MIN(wbc) AS wbc_min,
    MAX(wbc) AS wbc_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first
  FROM icustays AS ie
    LEFT JOIN complete_blood_count AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
  GROUP BY ie.stay_id
),
chem AS (
  SELECT ie.stay_id,
    MIN(albumin) AS albumin_min,
    MAX(albumin) AS albumin_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(globulin) AS globulin_min,
    MAX(globulin) AS globulin_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(total_protein) AS total_protein_min,
    MAX(total_protein) AS total_protein_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(aniongap) AS aniongap_min,
    MAX(aniongap) AS aniongap_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(bicarbonate) AS bicarbonate_min,
    MAX(bicarbonate) AS bicarbonate_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(bun) AS bun_min,
    MAX(bun) AS bun_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(calcium) AS calcium_min,
    MAX(calcium) AS calcium_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(chloride) AS chloride_min,
    MAX(chloride) AS chloride_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(creatinine) AS creatinine_min,
    MAX(creatinine) AS creatinine_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(glucose) AS glucose_min,
    MAX(glucose) AS glucose_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(sodium) AS sodium_min,
    MAX(sodium) AS sodium_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(potassium) AS potassium_min,
    MAX(potassium) AS potassium_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first
  FROM icustays AS ie
    LEFT JOIN chemistry AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
  GROUP BY ie.stay_id
),
diff AS (
  SELECT ie.stay_id,
    MIN(basophils_abs) AS abs_basophils_min,
    MAX(basophils_abs) AS abs_basophils_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(eosinophils_abs) AS abs_eosinophils_min,
    MAX(eosinophils_abs) AS abs_eosinophils_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(lymphocytes_abs) AS abs_lymphocytes_min,
    MAX(lymphocytes_abs) AS abs_lymphocytes_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(monocytes_abs) AS abs_monocytes_min,
    MAX(monocytes_abs) AS abs_monocytes_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(neutrophils_abs) AS abs_neutrophils_min,
    MAX(neutrophils_abs) AS abs_neutrophils_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(atypical_lymphocytes) AS atyps_min,
    MAX(atypical_lymphocytes) AS atyps_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(bands) AS bands_min,
    MAX(bands) AS bands_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(immature_granulocytes) AS imm_granulocytes_min,
    MAX(immature_granulocytes) AS imm_granulocytes_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(metamyelocytes) AS metas_min,
    MAX(metamyelocytes) AS metas_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(nrbc) AS nrbc_min,
    MAX(nrbc) AS nrbc_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first
  FROM icustays AS ie
    LEFT JOIN blood_differential AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
  GROUP BY ie.stay_id
),
coag AS (
  SELECT ie.stay_id,
    MIN(d_dimer) AS d_dimer_min,
    MAX(d_dimer) AS d_dimer_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(fibrinogen) AS fibrinogen_min,
    MAX(fibrinogen) AS fibrinogen_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(thrombin) AS thrombin_min,
    MAX(thrombin) AS thrombin_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(inr) AS inr_min,
    MAX(inr) AS inr_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(pt) AS pt_min,
    MAX(pt) AS pt_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(ptt) AS ptt_min,
    MAX(ptt) AS ptt_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first
  FROM icustays AS ie
    LEFT JOIN coagulation AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
  GROUP BY ie.stay_id
),
enz AS (
  SELECT ie.stay_id,
    MIN(alt) AS alt_min,
    MAX(alt) AS alt_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(alp) AS alp_min,
    MAX(alp) AS alp_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(ast) AS ast_min,
    MAX(ast) AS ast_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(amylase) AS amylase_min,
    MAX(amylase) AS amylase_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(bilirubin_total) AS bilirubin_total_min,
    MAX(bilirubin_total) AS bilirubin_total_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(bilirubin_direct) AS bilirubin_direct_min,
    MAX(bilirubin_direct) AS bilirubin_direct_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(bilirubin_indirect) AS bilirubin_indirect_min,
    MAX(bilirubin_indirect) AS bilirubin_indirect_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(ck_cpk) AS ck_cpk_min,
    MAX(ck_cpk) AS ck_cpk_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(ck_mb) AS ck_mb_min,
    MAX(ck_mb) AS ck_mb_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(ggt) AS ggt_min,
    MAX(ggt) AS ggt_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first,
    MIN(ld_ldh) AS ld_ldh_min,
    MAX(ld_ldh) AS ld_ldh_max,
    FIRST_VALUE(wbc) OVER (
      PARTITION BY stay_id
      ORDER BY charttime
    ) AS wbc_first
  FROM icustays AS ie
    LEFT JOIN enzyme AS le ON le.subject_id = ie.subject_id
    AND le.charttime >= ie.intime - INTERVAL '6 HOUR'
    AND le.charttime <= ie.intime + INTERVAL '1 DAY'
  GROUP BY ie.stay_id
)
SELECT ie.subject_id,
  ie.stay_id,
  /* complete blood count */
  hematocrit_min,
  hematocrit_max,
  hematocrit_first,
  hemoglobin_min,
  hemoglobin_max,
  hemoglobin_first,
  platelets_min,
  platelets_max,
  platelets_first,
  wbc_min,
  wbc_max,
  wbc_first,
  /* chemistry */
  albumin_min,
  albumin_max,
  albumin_first,
  globulin_min,
  globulin_max,
  globulin_first,
  total_protein_min,
  total_protein_max,
  total_protein_first,
  aniongap_min,
  aniongap_max,
  aniongap_first,
  bicarbonate_min,
  bicarbonate_max,
  bicarbonate_first,
  bun_min,
  bun_max,
  bun_first,
  calcium_min,
  calcium_max,
  calcium_first,
  chloride_min,
  chloride_max,
  chloride_first,
  creatinine_min,
  creatinine_max,
  creatinine_first,
  glucose_min,
  glucose_max,
  glucose_first,
  sodium_min,
  sodium_max,
  sodium_first,
  potassium_min,
  potassium_max,
  potassium_first,
  /* blood differential */
  abs_basophils_min,
  abs_basophils_max,
  abs_basophils_first,
  abs_eosinophils_min,
  abs_eosinophils_max,
  abs_eosinophils_first,
  abs_lymphocytes_min,
  abs_lymphocytes_max,
  abs_lymphocytes_first,
  abs_monocytes_min,
  abs_monocytes_max,
  abs_monocytes_first,
  abs_neutrophils_min,
  abs_neutrophils_max,
  abs_neutrophils_first,
  atyps_min,
  atyps_max,
  atyps_first,
  bands_min,
  bands_max,
  bands_first,
  imm_granulocytes_min,
  imm_granulocytes_max,
  imm_granulocytes_first,
  metas_min,
  metas_max,
  metas_first,
  nrbc_min,
  nrbc_max,
  nrbc_first,
  /* coagulation */
  d_dimer_min,
  d_dimer_max,
  d_dimer_first,
  fibrinogen_min,
  fibrinogen_max,
  fibrinogen_first,
  thrombin_min,
  thrombin_max,
  thrombin_first,
  inr_min,
  inr_max,
  inr_first,
  pt_min,
  pt_max,
  pt_first,
  ptt_min,
  ptt_max,
  ptt_first,
  /* enzymes and bilirubin */
  alt_min,
  alt_max,
  alt_first,
  alp_min,
  alp_max,
  alp_first,
  ast_min,
  ast_max,
  ast_first,
  amylase_min,
  amylase_max,
  amylase_first,
  bilirubin_total_min,
  bilirubin_total_max,
  bilirubin_total_first,
  bilirubin_direct_min,
  bilirubin_direct_max,
  bilirubin_direct_first,
  bilirubin_indirect_min,
  bilirubin_indirect_max,
  bilirubin_indirect_first,
  ck_cpk_min,
  ck_cpk_max,
  ck_cpk_first,
  ck_mb_min,
  ck_mb_max,
  ck_mb_first,
  ggt_min,
  ggt_max,
  ggt_first,
  ld_ldh_min,
  ld_ldh_max,
  ld_ldh_first
FROM icustays AS ie
  LEFT JOIN cbc ON ie.stay_id = cbc.stay_id
  LEFT JOIN chem ON ie.stay_id = chem.stay_id
  LEFT JOIN diff ON ie.stay_id = diff.stay_id
  LEFT JOIN coag ON ie.stay_id = coag.stay_id
  LEFT JOIN enz ON ie.stay_id = enz.stay_id