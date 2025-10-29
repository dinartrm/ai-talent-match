/* STEP 2 — TALENT MATCHING (TV/TGV) */

WITH
/* 0) Bobot TGV */
params AS (
  SELECT
    0.57::double precision AS w_competency,
    0.15::double precision AS w_strength,
    0.10::double precision AS w_cognitive,
    0.10::double precision AS w_behavior,
    0.08::double precision AS w_readiness
),

/* 1) Benchmark median Top Performer */
benchmark AS (
  SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "SEA") AS b_sea,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "IDS") AS b_ids,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "CEX") AS b_cex,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "VCU") AS b_vcu,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "CSI") AS b_csi,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cognitive_index) AS b_cognitive,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_P") AS b_p,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_W") AS b_w,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_I") AS b_i,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tenure_years) AS b_tenure,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY grade_num) AS b_grade,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY strength_match_score) AS b_strength
  FROM public.employees_scored
  WHERE is_top_performer = 1
),

/* 2) TV match (per variable) */
tv_match AS (
  /* Competency */
  SELECT e.employee_id, e.directorate, e."position" AS role, e.grade,
         'Competency'::text AS tgv_name, 'SEA'::text AS tv_name,
         b.b_sea AS baseline_score, e."SEA" AS user_score,
         CASE WHEN b.b_sea IS NULL OR b.b_sea=0 OR e."SEA" IS NULL
              THEN NULL ELSE LEAST(e."SEA"/b.b_sea,1.0)*100 END AS tv_match_rate
  FROM public.employees_scored e CROSS JOIN benchmark b

  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Competency','IDS', b.b_ids, e."IDS",
         CASE WHEN b.b_ids IS NULL OR b.b_ids=0 OR e."IDS" IS NULL
              THEN NULL ELSE LEAST(e."IDS"/b.b_ids,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Competency','CEX', b.b_cex, e."CEX",
         CASE WHEN b.b_cex IS NULL OR b.b_cex=0 OR e."CEX" IS NULL
              THEN NULL ELSE LEAST(e."CEX"/b.b_cex,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Competency','VCU', b.b_vcu, e."VCU",
         CASE WHEN b.b_vcu IS NULL OR b.b_vcu=0 OR e."VCU" IS NULL
              THEN NULL ELSE LEAST(e."VCU"/b.b_vcu,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Competency','CSI', b.b_csi, e."CSI",
         CASE WHEN b.b_csi IS NULL OR b.b_csi=0 OR e."CSI" IS NULL
              THEN NULL ELSE LEAST(e."CSI"/b.b_csi,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  /* Strength DNA */
  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Strength DNA','Strength_Match', b.b_strength, e.strength_match_score,
         CASE WHEN b.b_strength IS NULL OR b.b_strength=0 OR e.strength_match_score IS NULL
              THEN NULL ELSE LEAST(e.strength_match_score/b.b_strength,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  /* Cognitive */
  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Cognitive','Cognitive_Index', b.b_cognitive, e.cognitive_index,
         CASE WHEN b.b_cognitive IS NULL OR b.b_cognitive=0 OR e.cognitive_index IS NULL
              THEN NULL ELSE LEAST(e.cognitive_index/b.b_cognitive,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  /* Behavioral */
  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Behavior','PAPI_P', b.b_p, e."Papi_P",
         CASE WHEN b.b_p IS NULL OR b.b_p=0 OR e."Papi_P" IS NULL
              THEN NULL ELSE LEAST(e."Papi_P"/b.b_p,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Behavior','PAPI_W', b.b_w, e."Papi_W",
         CASE WHEN b.b_w IS NULL OR b.b_w=0 OR e."Papi_W" IS NULL
              THEN NULL ELSE LEAST(e."Papi_W"/b.b_w,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Behavior','PAPI_I', b.b_i, e."Papi_I",
         CASE WHEN b.b_i IS NULL OR b.b_i=0 OR e."Papi_I" IS NULL
              THEN NULL ELSE LEAST(e."Papi_I"/b.b_i,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  /* Readiness */
  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Readiness','Tenure_Years', b.b_tenure, e.tenure_years,
         CASE WHEN b.b_tenure IS NULL OR b.b_tenure=0 OR e.tenure_years IS NULL
              THEN NULL ELSE LEAST(e.tenure_years/b.b_tenure,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b

  UNION ALL
  SELECT e.employee_id, e.directorate, e."position", e.grade,
         'Readiness','Grade', b.b_grade, e.grade_num,
         CASE WHEN b.b_grade IS NULL OR b.b_grade=0 OR e.grade_num IS NULL
              THEN NULL ELSE LEAST(e.grade_num/b.b_grade,1.0)*100 END
  FROM public.employees_scored e CROSS JOIN benchmark b
),

/* 3) TGV average (per group per employee) */
tgv_match AS (
  SELECT
    employee_id, directorate, role, grade, tgv_name,
    AVG(tv_match_rate) AS tgv_match_rate
  FROM tv_match
  GROUP BY employee_id, directorate, role, grade, tgv_name
),

/* 4) Pivot TGV ke kolom (tanpa bobot) */
final_rates AS (
  SELECT
    t.employee_id, t.directorate, t.role, t.grade,
    MAX(CASE WHEN tgv_name='Competency'   THEN tgv_match_rate END) AS comp_rate,
    MAX(CASE WHEN tgv_name='Strength DNA' THEN tgv_match_rate END) AS strength_rate,
    MAX(CASE WHEN tgv_name='Cognitive'    THEN tgv_match_rate END) AS cognitive_rate,
    MAX(CASE WHEN tgv_name='Behavior'     THEN tgv_match_rate END) AS behavior_rate,
    MAX(CASE WHEN tgv_name='Readiness'    THEN tgv_match_rate END) AS readiness_rate
  FROM tgv_match t
  GROUP BY t.employee_id, t.directorate, t.role, t.grade
),

/* 5) Hitung final berbobot (no GROUP BY) */
final_match AS (
  SELECT
    r.employee_id, r.directorate, r.role, r.grade,
    r.comp_rate, r.strength_rate, r.cognitive_rate, r.behavior_rate, r.readiness_rate,
    ( COALESCE(r.comp_rate,0)     * p.w_competency
    + COALESCE(r.strength_rate,0) * p.w_strength
    + COALESCE(r.cognitive_rate,0)* p.w_cognitive
    + COALESCE(r.behavior_rate,0) * p.w_behavior
    + COALESCE(r.readiness_rate,0)* p.w_readiness
    ) AS final_match_rate
  FROM final_rates r
  CROSS JOIN params p
)

/* 6) OUTPUT TV-level (sesuai brief) */
SELECT
  d.employee_id,
  d.directorate,
  d.role,
  d.grade,
  d.tgv_name,
  d.tv_name,
  ROUND(d.baseline_score::numeric,2) AS baseline_score,
  ROUND(d.user_score::numeric,2)     AS user_score,
  ROUND(d.tv_match_rate::numeric,2)  AS tv_match_rate,
  ROUND(t.tgv_match_rate::numeric,2) AS tgv_match_rate,
  ROUND(f.final_match_rate::numeric,2) AS final_match_rate
FROM tv_match d
LEFT JOIN tgv_match t
  ON t.employee_id=d.employee_id AND t.tgv_name=d.tgv_name
LEFT JOIN final_match f
  ON f.employee_id=d.employee_id
ORDER BY f.final_match_rate DESC, d.employee_id, d.tgv_name, d.tv_name
LIMIT 500;