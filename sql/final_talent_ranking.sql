WITH params AS (
  SELECT 0.57::float8 w_competency,0.15 w_strength,0.10 w_cognitive,0.10 w_behavior,0.08 w_readiness
),
benchmark AS (
  SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "SEA") b_sea,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "IDS") b_ids,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "CEX") b_cex,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "VCU") b_vcu,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "CSI") b_csi,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cognitive_index) b_cognitive,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_P") b_p,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_W") b_w,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_I") b_i,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tenure_years) b_tenure,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY grade_num) b_grade,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY strength_match_score) b_strength
  FROM public.employees_scored WHERE is_top_performer=1
),
tv_match AS (
  SELECT e.employee_id, e.directorate, e."position" AS role, e.grade, 'Competency' tgv_name, 'SEA' tv_name,
         LEAST(e."SEA"/NULLIF(b.b_sea,0),1.0)*100 tv_match_rate
  FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Competency','IDS',LEAST(e."IDS"/NULLIF(b.b_ids,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Competency','CEX',LEAST(e."CEX"/NULLIF(b.b_cex,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Competency','VCU',LEAST(e."VCU"/NULLIF(b.b_vcu,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Competency','CSI',LEAST(e."CSI"/NULLIF(b.b_csi,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Strength DNA','Strength_Match',LEAST(e.strength_match_score/NULLIF(b.b_strength,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Cognitive','Cognitive_Index',LEAST(e.cognitive_index/NULLIF(b.b_cognitive,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Behavior','PAPI_P',LEAST(e."Papi_P"/NULLIF(b.b_p,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Behavior','PAPI_W',LEAST(e."Papi_W"/NULLIF(b.b_w,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Behavior','PAPI_I',LEAST(e."Papi_I"/NULLIF(b.b_i,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Readiness','Tenure_Years',LEAST(e.tenure_years/NULLIF(b.b_tenure,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
  UNION ALL SELECT e.employee_id, e.directorate, e."position", e.grade,'Readiness','Grade',LEAST(e.grade_num/NULLIF(b.b_grade,0),1.0)*100 FROM public.employees_scored e CROSS JOIN benchmark b
),
tgv_match AS (
  SELECT employee_id, directorate, role, grade, tgv_name, AVG(tv_match_rate) tgv_match_rate
  FROM tv_match GROUP BY employee_id, directorate, role, grade, tgv_name
),
final_rates AS (
  SELECT employee_id,directorate,role,grade,
         MAX(CASE WHEN tgv_name='Competency' THEN tgv_match_rate END) comp_rate,
         MAX(CASE WHEN tgv_name='Strength DNA' THEN tgv_match_rate END) strength_rate,
         MAX(CASE WHEN tgv_name='Cognitive' THEN tgv_match_rate END) cognitive_rate,
         MAX(CASE WHEN tgv_name='Behavior' THEN tgv_match_rate END) behavior_rate,
         MAX(CASE WHEN tgv_name='Readiness' THEN tgv_match_rate END) readiness_rate
  FROM tgv_match GROUP BY employee_id,directorate,role,grade
)
SELECT employee_id, directorate, role, grade,
       ROUND(comp_rate::numeric,2)       AS comp_rate,
       ROUND(strength_rate::numeric,2)   AS strength_rate,
       ROUND(cognitive_rate::numeric,2)  AS cognitive_rate,
       ROUND(behavior_rate::numeric,2)   AS behavior_rate,
       ROUND(readiness_rate::numeric,2)  AS readiness_rate,
       ROUND( (comp_rate*0.57
             + strength_rate*0.15
             + cognitive_rate*0.10
             + behavior_rate*0.10
             + readiness_rate*0.08)::numeric, 2) AS final_match_rate
FROM final_rates
ORDER BY final_match_rate DESC
LIMIT 50;