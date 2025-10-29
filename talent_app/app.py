
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
import requests

# --------------------------------------------------
# Session state guards (prevent UI resets)
# --------------------------------------------------
if "df_rank" not in st.session_state:
    st.session_state["df_rank"] = None
if "bench_ids" not in st.session_state:
    st.session_state["bench_ids"] = []
if "resolved_grade" not in st.session_state:
    st.session_state["resolved_grade"] = None
if "bench_note" not in st.session_state:
    st.session_state["bench_note"] = ""
if "compare_ids" not in st.session_state:
    st.session_state["compare_ids"] = []

# --------------------------------------------------
# THEME — single hue green
# --------------------------------------------------
THEME = {
    "base": "#10B981",     # green-500
    "base_soft": "#34D399",# green-400
    "base_dim": "#059669", # green-600
    "accent": "#22C55E",   # green mid
    "bg_band": "#064E3B",  # dark
    "neg": "#B91C1C"       # red
}

st.set_page_config(page_title="AI Talent Match & Storytelling", layout="wide")

# --------------------------------------------------
# Secrets / DB / API
# --------------------------------------------------
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# --------------------------------------------------
# Helpers: basic pulls
# --------------------------------------------------
def get_all_employees():
    q = """
        SELECT employee_id, fullname, directorate, division, department, "position", grade
        FROM public.employees_scored
        ORDER BY fullname
    """
    return pd.read_sql(q, engine)

def get_role_suggestions():
    q = 'SELECT DISTINCT "position" AS role FROM public.employees_scored ORDER BY 1'
    try:
        df = pd.read_sql(q, engine)
        return df["role"].dropna().tolist()
    except Exception:
        return []

# --------------------------------------------------
# TVs and TGVs mapping
# --------------------------------------------------
COMP_TVS = ["SEA","IDS","CEX","VCU","CSI"]
BEH_TVS  = ["Papi_P","Papi_W","Papi_I"]
COG_TVS  = ["cognitive_index"]
STR_TVS  = ["strength_match_score"]
READ_TVS = ["tenure_years","grade_num"]
ALL_TVS  = COMP_TVS + BEH_TVS + COG_TVS + STR_TVS + READ_TVS
TV_TO_TGV = {tv:"Competency" for tv in COMP_TVS} | {tv:"Behavior" for tv in BEH_TVS} | \
            {tv:"Cognitive" for tv in COG_TVS} | {tv:"Strength DNA" for tv in STR_TVS} | \
            {tv:"Readiness" for tv in READ_TVS}

# --------------------------------------------------
# Benchmark resolution (Rule A)
# 1) If user picks n<5 employees -> expand to all Top Performers of same grade
# 2) If still <5 -> expand to company Top Performers (all grades)
# Returns: (resolved_selected_ids, resolved_grade, note)
# --------------------------------------------------
def resolve_benchmark(selected_ids, ui_grade):
    """
    ui_grade: 'III' / 'IV' / 'V' / None
    """
    # Helper to count cohort size for a condition
    def cohort_size(cond_sql, params):
        q = f"SELECT COUNT(*) FROM public.employees_scored WHERE {cond_sql}"
        return int(pd.read_sql(text(q), engine, params=params).iloc[0,0])

    # Case 0 — no user benchmark: use Top Performers filtered by grade (if provided)
    if not selected_ids:
        # Count top performers in grade (if grade provided)
        if ui_grade:
            n = cohort_size('is_top_performer=1 AND grade = :g', {"g": ui_grade})
            if n >= 5:
                return (None, ui_grade, f"Benchmark = Top Performers (grade {ui_grade})")
        # Fallback to company-wide top performers
        n2 = cohort_size('is_top_performer=1', {})
        return (None, None, "Benchmark = Company Top Performers")

    # Case 1 — user provided selected employees as benchmark
    # How many of them exist in DB (sanity)
    n_sel = cohort_size('employee_id = ANY(:ids)', {"ids": selected_ids})
    if n_sel >= 5:
        return (selected_ids, ui_grade, f"Benchmark = Selected employees ({n_sel})")

    # Expand to grade-level top performers if grade given and sufficient
    if ui_grade:
        n_grade = cohort_size('is_top_performer=1 AND grade = :g', {"g": ui_grade})
        if n_grade >= 5:
            return (None, ui_grade, f"Benchmark = Top Performers (grade {ui_grade})")

    # Final fallback: company top performers
    return (None, None, "Benchmark = Company Top Performers")

# --------------------------------------------------
# Core SQL — builds ranking using resolved benchmark and grade pool
# --------------------------------------------------
def build_sql(resolved_selected_ids, resolved_grade):
    where_bench = "employee_id = ANY(:selected_ids)" if resolved_selected_ids else "is_top_performer = 1"
    pool_filter = "( :job_level IS NULL OR grade = :job_level )"

    tv_sql = f"""
    WITH
    params AS (
      SELECT
        0.57::double precision AS w_competency,
        0.15::double precision AS w_strength,
        0.10::double precision AS w_cognitive,
        0.10::double precision AS w_behavior,
        0.08::double precision AS w_readiness
    ),
    pool AS (
      SELECT *
      FROM public.employees_scored
      WHERE {pool_filter}
    ),
    benchmark AS (
      SELECT
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "SEA"::float)              AS b_sea,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "IDS"::float)              AS b_ids,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "CEX"::float)              AS b_cex,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "VCU"::float)              AS b_vcu,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "CSI"::float)              AS b_csi,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cognitive_index::float)    AS b_cognitive,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_P"::float)           AS b_p,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_W"::float)           AS b_w,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Papi_I"::float)           AS b_i,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tenure_years::float)       AS b_tenure,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY grade_num::float)          AS b_grade,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY strength_match_score::float) AS b_strength
      FROM pool
      WHERE {where_bench}
    ),
    tv_match AS (
      -- COMPETENCY TVs
      SELECT e.employee_id, e.fullname, e.directorate, e."position" AS role, e.grade,
             'Competency'::text AS tgv_name, 'SEA'::text AS tv_name,
             b.b_sea AS baseline_score, e."SEA" AS user_score,
             CASE
               WHEN b.b_sea IS NULL OR e."SEA" IS NULL THEN NULL
               ELSE LEAST( e."SEA" / NULLIF(b.b_sea,0), 1.0) * 100
             END AS tv_match_rate
      FROM pool e CROSS JOIN benchmark b

      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Competency','IDS', b.b_ids, e."IDS",
             CASE WHEN b.b_ids IS NULL OR e."IDS" IS NULL THEN NULL
                  ELSE LEAST( e."IDS" / NULLIF(b.b_ids,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Competency','CEX', b.b_cex, e."CEX",
             CASE WHEN b.b_cex IS NULL OR e."CEX" IS NULL THEN NULL
                  ELSE LEAST( e."CEX" / NULLIF(b.b_cex,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Competency','VCU', b.b_vcu, e."VCU",
             CASE WHEN b.b_vcu IS NULL OR e."VCU" IS NULL THEN NULL
                  ELSE LEAST( e."VCU" / NULLIF(b.b_vcu,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Competency','CSI', b.b_csi, e."CSI",
             CASE WHEN b.b_csi IS NULL OR e."CSI" IS NULL THEN NULL
                  ELSE LEAST( e."CSI" / NULLIF(b.b_csi,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      -- STRENGTH DNA
      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Strength DNA','Strength_Match', b.b_strength, e.strength_match_score,
             CASE WHEN b.b_strength IS NULL OR e.strength_match_score IS NULL THEN NULL
                  ELSE LEAST( e.strength_match_score / NULLIF(b.b_strength,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      -- COGNITIVE
      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Cognitive','Cognitive_Index', b.b_cognitive, e.cognitive_index,
             CASE WHEN b.b_cognitive IS NULL OR e.cognitive_index IS NULL THEN NULL
                  ELSE LEAST( e.cognitive_index / NULLIF(b.b_cognitive,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      -- BEHAVIOR
      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Behavior','PAPI_P', b.b_p, e."Papi_P",
             CASE WHEN b.b_p IS NULL OR e."Papi_P" IS NULL THEN NULL
                  ELSE LEAST( e."Papi_P" / NULLIF(b.b_p,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Behavior','PAPI_W', b.b_w, e."Papi_W",
             CASE WHEN b.b_w IS NULL OR e."Papi_W" IS NULL THEN NULL
                  ELSE LEAST( e."Papi_W" / NULLIF(b.b_w,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Behavior','PAPI_I', b.b_i, e."Papi_I",
             CASE WHEN b.b_i IS NULL OR e."Papi_I" IS NULL THEN NULL
                  ELSE LEAST( e."Papi_I" / NULLIF(b.b_i,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      -- READINESS
      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Readiness','Tenure_Years', b.b_tenure, e.tenure_years,
             CASE WHEN b.b_tenure IS NULL OR e.tenure_years IS NULL THEN NULL
                  ELSE LEAST( e.tenure_years / NULLIF(b.b_tenure,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b

      UNION ALL
      SELECT e.employee_id, e.fullname, e.directorate, e."position", e.grade,
             'Readiness','Grade', b.b_grade, e.grade_num,
             CASE WHEN b.b_grade IS NULL OR e.grade_num IS NULL THEN NULL
                  ELSE LEAST( e.grade_num / NULLIF(b.b_grade,0), 1.0) * 100 END
      FROM pool e CROSS JOIN benchmark b
    ),
    tgv_match AS (
      SELECT employee_id, fullname, directorate, role, grade, tgv_name,
             AVG(tv_match_rate) AS tgv_match_rate
      FROM tv_match
      GROUP BY employee_id, fullname, directorate, role, grade, tgv_name
    ),
    final_rates AS (
      SELECT employee_id, fullname, directorate, role, grade,
        MAX(CASE WHEN tgv_name='Competency'   THEN tgv_match_rate END) AS comp_rate,
        MAX(CASE WHEN tgv_name='Strength DNA' THEN tgv_match_rate END) AS strength_rate,
        MAX(CASE WHEN tgv_name='Cognitive'    THEN tgv_match_rate END) AS cognitive_rate,
        MAX(CASE WHEN tgv_name='Behavior'     THEN tgv_match_rate END) AS behavior_rate,
        MAX(CASE WHEN tgv_name='Readiness'    THEN tgv_match_rate END) AS readiness_rate
      FROM tgv_match
      GROUP BY employee_id, fullname, directorate, role, grade
    )
    SELECT
      f.employee_id, f.fullname, f.directorate, f.role, f.grade,
      ROUND(f.comp_rate::numeric,2)      AS comp_rate,
      ROUND(f.strength_rate::numeric,2)  AS strength_rate,
      ROUND(f.cognitive_rate::numeric,2) AS cognitive_rate,
      ROUND(f.behavior_rate::numeric,2)  AS behavior_rate,
      ROUND(f.readiness_rate::numeric,2) AS readiness_rate,
      ROUND(
        CASE
          WHEN (
            (CASE WHEN f.comp_rate      IS NOT NULL THEN 0.57 ELSE 0 END) +
            (CASE WHEN f.strength_rate  IS NOT NULL THEN 0.15 ELSE 0 END) +
            (CASE WHEN f.cognitive_rate IS NOT NULL THEN 0.10 ELSE 0 END) +
            (CASE WHEN f.behavior_rate  IS NOT NULL THEN 0.10 ELSE 0 END) +
            (CASE WHEN f.readiness_rate IS NOT NULL THEN 0.08 ELSE 0 END)
          ) = 0 THEN NULL
          ELSE
            (
              COALESCE(f.comp_rate,0)      * 0.57 +
              COALESCE(f.strength_rate,0)  * 0.15 +
              COALESCE(f.cognitive_rate,0) * 0.10 +
              COALESCE(f.behavior_rate,0)  * 0.10 +
              COALESCE(f.readiness_rate,0) * 0.08
            )
            /
            (
              (CASE WHEN f.comp_rate      IS NOT NULL THEN 0.57 ELSE 0 END) +
              (CASE WHEN f.strength_rate  IS NOT NULL THEN 0.15 ELSE 0 END) +
              (CASE WHEN f.cognitive_rate IS NOT NULL THEN 0.10 ELSE 0 END) +
              (CASE WHEN f.behavior_rate  IS NOT NULL THEN 0.10 ELSE 0 END) +
              (CASE WHEN f.readiness_rate IS NOT NULL THEN 0.08 ELSE 0 END)
            )
        END
      ::numeric, 2) AS final_match_rate
    FROM final_rates f
    ORDER BY final_match_rate DESC NULLS LAST;
    """
    return tv_sql

def run_sql(resolved_selected_ids, resolved_grade):
    sql = build_sql(resolved_selected_ids, resolved_grade if resolved_grade not in (None, "All") else None)
    params = {}
    if resolved_selected_ids:
        params["selected_ids"] = resolved_selected_ids
    params["job_level"] = None if (resolved_grade in (None, "All")) else resolved_grade
    return pd.read_sql(text(sql), engine, params=params)

# --------------------------------------------------
# Extra pulls for TVs & Strengths
# --------------------------------------------------
def get_benchmark_cohort_for_median(resolved_selected_ids, resolved_grade):
    # aligns with SQL benchmark scope
    where = []
    params = {}
    if resolved_selected_ids:
        where.append('employee_id = ANY(:selected_ids)')
        params["selected_ids"] = resolved_selected_ids
    else:
        where.append('is_top_performer = 1')
    if resolved_grade and resolved_grade != "All":
        where.append('grade = :job_level')
        params["job_level"] = resolved_grade
    cond = " AND ".join(where) if where else "TRUE"

    q = f"""
    SELECT employee_id, fullname, grade, directorate, "position",
           "SEA","IDS","CEX","VCU","CSI",
           "Papi_P","Papi_W","Papi_I",
           cognitive_index, strength_match_score,
           tenure_years, grade_num,
           strength_1, strength_2, strength_3, strength_4, strength_5
    FROM public.employees_scored
    WHERE {cond}
    """
    return pd.read_sql(text(q), engine, params=params)

def get_candidates_tv(candidate_ids):
    if not candidate_ids:
        return pd.DataFrame()
    q = """
    SELECT employee_id, fullname, grade, directorate, "position",
           "SEA","IDS","CEX","VCU","CSI",
           "Papi_P","Papi_W","Papi_I",
           cognitive_index, strength_match_score,
           tenure_years, grade_num,
           strength_1, strength_2, strength_3, strength_4, strength_5
    FROM public.employees_scored
    WHERE employee_id = ANY(:ids)
    """
    return pd.read_sql(text(q), engine, params={"ids": candidate_ids})

def median_or_nan(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return np.nan
    return float(s.median())

def safe(x):
    try:
        if pd.isna(x): return 0.0
        return float(x)
    except Exception:
        return 0.0

# --------------------------------------------------
# Visual helpers
# --------------------------------------------------
def plot_radar_multi(cand_rows, bench_tgv_avg):
    cats = ["Competency","Strength DNA","Cognitive","Behavior","Readiness"]
    fig = go.Figure()
    bench_vals = [safe(bench_tgv_avg.get("comp_rate",100)),
                  safe(bench_tgv_avg.get("strength_rate",100)),
                  safe(bench_tgv_avg.get("cognitive_rate",100)),
                  safe(bench_tgv_avg.get("behavior_rate",100)),
                  safe(bench_tgv_avg.get("readiness_rate",100))]
    fig.add_trace(go.Scatterpolar(
        r=bench_vals + [bench_vals[0]],
        theta=cats + [cats[0]], fill='toself', name='Benchmark',
        line=dict(color=THEME["base_soft"]), fillcolor='rgba(52,211,153,0.15)'
    ))
    palette = [THEME["base_dim"], THEME["accent"], THEME["base"]]
    for idx, row in enumerate(cand_rows[:3]):
        vals = [safe(row.get("comp_rate")), safe(row.get("strength_rate")),
                safe(row.get("cognitive_rate")), safe(row.get("behavior_rate")),
                safe(row.get("readiness_rate"))]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]], fill='toself',
            name=row.get("fullname","Candidate"),
            line=dict(color=palette[idx % len(palette)], width=2),
            fillcolor='rgba(16,185,129,0.20)' if idx==0 else None
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=True, height=480)
    return fig

def plot_gap_tgv_bar(candidate_row, bench_tgv_avg):
    labels = ["Competency","Strength","Cognitive","Behavior","Readiness"]
    gaps = [
        safe(candidate_row.get("comp_rate"))      - safe(bench_tgv_avg.get("comp_rate",100)),
        safe(candidate_row.get("strength_rate"))  - safe(bench_tgv_avg.get("strength_rate",100)),
        safe(candidate_row.get("cognitive_rate")) - safe(bench_tgv_avg.get("cognitive_rate",100)),
        safe(candidate_row.get("behavior_rate"))  - safe(bench_tgv_avg.get("behavior_rate",100)),
        safe(candidate_row.get("readiness_rate")) - safe(bench_tgv_avg.get("readiness_rate",100)),
    ]
    colors = [THEME["accent"] if g>=0 else THEME["neg"] for g in gaps]
    fig = go.Figure(go.Bar(x=labels, y=gaps, marker_color=colors,
                           text=[f"{g:+.1f}%" for g in gaps], textposition="outside"))
    fig.add_hline(y=0, line_width=1, line_color="#94A3B8")
    ymax = max(20, np.nanmax(np.abs(gaps)))
    ymax = min(max(ymax, 20), 100)
    fig.update_layout(height=330, yaxis_title="Gap vs Benchmark (%)",
                      yaxis=dict(range=[-ymax, ymax]))
    return fig

def compute_tv_match_percent(val, bench):
    if bench is None or pd.isna(bench) or bench == 0 or pd.isna(val):
        return np.nan
    return min(val/bench, 1.0) * 100.0

def build_tv_match_heatmap(cands_tv_df, bench_tv_median, title="TV Match vs Benchmark (Capped at 100%)"):
    if cands_tv_df.empty:
        return go.Figure()
    tvs = ALL_TVS
    cand_labels = [f"{r.employee_id} — {r.fullname}" for _, r in cands_tv_df.iterrows()]
    data = []
    for tv in tvs:
        row_vals = []
        bench = bench_tv_median.get(tv, np.nan)
        for _, r in cands_tv_df.iterrows():
            row_vals.append(compute_tv_match_percent(safe(r.get(tv)), bench))
        data.append(row_vals)
    z = np.array(data, dtype=float)
    z_display = np.nan_to_num(z, nan=0.0)
    text = np.where(np.isnan(z), '-', (np.round(z,1)).astype(str))
    fig = go.Figure(data=go.Heatmap(
        z=z_display, x=cand_labels, y=tvs,
        colorscale=[[0, '#E8F5E9'], [1, THEME["base_soft"]]],
        colorbar=dict(title="Match %"),
        text=text, texttemplate="%{text}",
        hovertemplate="TV: %{y}<br>Candidate: %{x}<br>Match: %{z:.1f}%"
    ))
    fig.update_layout(title=title, height=520)
    return fig

def build_tv_gap_bar(candidate_tv_row, bench_tv_median, group="Competency"):
    tvs = [tv for tv in ALL_TVS if TV_TO_TGV[tv] == group]
    labels, gaps = [], []
    for tv in tvs:
        bench = bench_tv_median.get(tv, np.nan)
        match = compute_tv_match_percent(safe(candidate_tv_row.get(tv)), bench)
        gap = (match - 100.0) if not pd.isna(match) else 0.0
        labels.append(tv)
        gaps.append(gap)
    colors = [THEME["accent"] if g>=0 else THEME["neg"] for g in gaps]
    fig = go.Figure(go.Bar(x=labels, y=gaps, marker_color=colors,
                           text=[f"{g:+.1f}%" for g in gaps], textposition="outside"))
    fig.add_hline(y=0, line_width=1, line_color="#94A3B8")
    fig.update_layout(height=300, yaxis_title=f"Gap vs Benchmark ({group})")
    return fig

def strengths_frequency(tv_df, top_n=20):
    if tv_df.empty:
        return pd.Series(dtype=int)
    cols = ["strength_1","strength_2","strength_3","strength_4","strength_5"]
    s = tv_df[cols].stack().dropna()
    return s.value_counts().head(15)

def strengths_gap_for_candidate(candidate_tv_row, bench_df):
    c_strengths = [candidate_tv_row.get(f"strength_{i}") for i in range(1,6)]
    c_strengths = [s for s in c_strengths if pd.notna(s)]
    if not len(c_strengths):
        return pd.DataFrame(columns=["strength","benchmark_%"])
    if bench_df.empty:
        return pd.DataFrame({"strength": c_strengths, "benchmark_%": [0]*len(c_strengths)})
    cols = ["strength_1","strength_2","strength_3","strength_4","strength_5"]
    rows = []
    for stg in c_strengths:
        present = ((bench_df[cols] == stg).any(axis=1)).mean() * 100.0
        rows.append({"strength": stg, "benchmark_%": present})
    df = pd.DataFrame(rows).sort_values("benchmark_%", ascending=False)
    return df

# --------------------------------------------------
# AI helper via OpenRouter (requests)
# --------------------------------------------------
def call_ai(role_name, job_level, role_purpose, top_rows):
    bullets = []
    for _, r in top_rows.iterrows():
        bullets.append(
            f"- {r['fullname']} ({r['employee_id']}), Final {safe(r['final_match_rate']):.1f}%, "
            f"Comp {safe(r['comp_rate']):.1f}%, Strength {safe(r['strength_rate']):.1f}%, "
            f"Cog {safe(r['cognitive_rate']):.1f}%, Beh {safe(r['behavior_rate']):.1f}%, "
            f"Readiness {safe(r['readiness_rate']):.1f}%"
        )
    people_snippet = "\n".join(bullets) if bullets else "- (no candidates)"

    prompt = f"""
You are a Talent Intelligence assistant writing in concise consulting English.
Keep tone professional. Use bullet lists (each item on a new line).

ROLE CONTEXT
• Role Name: {role_name}
• Job Level: {job_level}
• Role Purpose (short): {role_purpose}

TOP CANDIDATES (short)
{people_snippet}

REQUIRED OUTPUT (MARKDOWN):

## Job Requirements
- Provide 6–10 bullet points (one per line).
- Focus on scope, KPIs, decision rights, business impact.

## Job Description
Write 3–5 short sentences (one paragraph). Plain, no bullets.

## Key Competencies (with Tools)
- Provide 5–9 items (one per line).
- Each bullet should pair capability and tool if relevant (e.g., SQL; BI dashboards in Power BI/Tableau; Python/pandas; dbt; Git; Airflow).
- Tie capabilities to business value when relevant (e.g., metrics layer, automated reporting, stakeholder storytelling).

## Why Top Candidates Rank Highest
- Provide 2–4 bullets referencing TGV categories (Competency / Strength DNA / Cognitive / Behavior / Readiness).
""".strip()

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "anthropic/claude-3-haiku",
        "messages": [
            {"role": "system", "content": "You are a talent intelligence assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content or "_AI returned no content._"
    except requests.HTTPError as e:
        return f"_HTTP error from OpenRouter: {e}\nBody: {resp.text}_"
    except Exception as e:
        return f"_Unexpected error calling AI: {e}_"

# --------------------------------------------------
# UI — Sidebar
# --------------------------------------------------
st.title("AI Talent Match & Storytelling")

with st.sidebar:
    st.subheader("Role Inputs")

    roles = get_role_suggestions()
    role_name = st.text_input(
        "Role name",
        value=(roles[0] if roles else "Brand Executive"),
        help="Enter target role name (free-text, not case-sensitive). Used for AI narrative only."
    )
    job_level_ui = st.selectbox("Job level", ["All","Junior","Middle","Senior"], index=2)

    grade_map = {
        "All": None,
        "Junior": "III",
        "Middle": "IV",
        "Senior": "V"
    }
    ui_grade = grade_map.get(job_level_ui)

    role_purpose = st.text_area(
        "Role purpose",
        value="Drive brand visibility and engagement through cross-channel marketing initiatives",
        help="Describe the core purpose of the role in 1–2 sentences. Guides the AI narrative."
    )
    st.markdown("---")

    df_emp = get_all_employees()
    sel = st.multiselect(
        "Pick Benchmark Employees (optional)",
        options=df_emp["employee_id"].tolist(),
        format_func=lambda x: f"{x} — {df_emp.loc[df_emp['employee_id']==x,'fullname'].values[0]}",
        help="Recommended: 5–15 proven performers in a similar role. The system auto-expands if your selection is too small."
    )
    run_btn = st.button("RUN Matching")

# =============================================
# ✅ Recommended Benchmark - Always Suggest if Empty
# =============================================
rec_bench = None
rec_count = 0

if job_level_ui != "All":
    db_grade = grade_map.get(job_level_ui)
    q = """
        SELECT employee_id
        FROM public.employees_scored
        WHERE is_top_performer = 1
        AND grade = :grade
    """
    res = pd.read_sql(text(q), engine, params={"grade": db_grade})
    if len(res) >= 5:
        rec_bench = res["employee_id"].tolist()
        rec_count = len(res)

# ✅ Recommend only if user has NOT selected benchmark yet
if not sel and rec_bench:
    with st.expander("Recommended Benchmark Found ✅", expanded=True):
        st.write(f"**Top Performers — Grade {db_grade}**")
        st.write(f"{rec_count} employees suitable as benchmark for this job level.")
        st.write("Recommended to improve comparison quality.")
        if st.button("Use Recommended Benchmark"):
            sel = rec_bench.copy()
            st.session_state["bench_ids"] = rec_bench
            st.rerun()

# --------------------------------------------------
# UI — Main flow
# --------------------------------------------------
if run_btn:
    resolved_ids = sel if sel else rec_bench  # Rule A apply
    resolved_grade = db_grade  # None if "All"

    with st.spinner("Computing match rates and building ranking..."):
        df_rank = run_sql(resolved_ids, resolved_grade)
        
        # ✅ Sanitize & sort
        _cols = ["final_match_rate","comp_rate","strength_rate",
                 "cognitive_rate","behavior_rate","readiness_rate"]
        for c in _cols:
            if c in df_rank.columns:
                df_rank[c] = pd.to_numeric(df_rank[c], errors="coerce").fillna(0.0)

        df_rank = df_rank.sort_values("final_match_rate", ascending=False)

    st.session_state["df_rank"] = df_rank
    st.session_state["bench_ids"] = resolved_ids
    st.session_state["resolved_grade"] = resolved_grade

else:
    df_rank = st.session_state.get("df_rank", None)

#  Guard
if df_rank is None or df_rank.empty:
    st.info("Fill **Role Inputs**, optionally pick Benchmark, then click **RUN Matching** to generate insights.")
    st.stop()

# --------------------------------------------------
# Benchmark objects for visuals
# --------------------------------------------------
bench_ids = st.session_state.get("bench_ids", [])
resolved_grade = st.session_state.get("resolved_grade", None)
bench_note = st.session_state.get("bench_note", "")

bench_df = get_benchmark_cohort_for_median(bench_ids if len(bench_ids)>0 else None, resolved_grade)
bench_tv_median = {tv: median_or_nan(bench_df[tv]) if tv in bench_df.columns else np.nan for tv in ALL_TVS}
bench_tgv_avg = {"comp_rate":100, "strength_rate":100, "cognitive_rate":100, "behavior_rate":100, "readiness_rate":100}

# --------------------------------------------------
# Summary Cards
# --------------------------------------------------
top = df_rank.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Total Candidates", len(df_rank))
c2.metric("Average Final Match", f"{df_rank['final_match_rate'].mean():.1f}%")
c3.metric("Top Match", f"{safe(top['final_match_rate']):.1f}%")

st.info(f"{bench_note} — Candidate pool filtered by job level: {resolved_grade if resolved_grade else 'All levels'}.")

# --------------------------------------------------
# Ranked Table
# --------------------------------------------------
st.markdown("### Ranked Candidates (Top 50)")
st.dataframe(df_rank.head(50), use_container_width=True)

# --------------------------------------------------
# Radar — Benchmark vs Top 3
# --------------------------------------------------
st.markdown("### Benchmark vs Candidates — Radar (Top 3)")
fig_radar = plot_radar_multi(df_rank.head(3).to_dict(orient="records"), bench_tgv_avg)
st.plotly_chart(fig_radar, use_container_width=True)

# --------------------------------------------------
# TGV Gap — Top Candidate vs Benchmark
# --------------------------------------------------
st.markdown("### TGV Gap — Top Candidate vs Benchmark")
fig_gap_tgv = plot_gap_tgv_bar(top, bench_tgv_avg)
st.plotly_chart(fig_gap_tgv, use_container_width=True)

# --------------------------------------------------
# Heatmap — TV Match vs Benchmark (Top 3)
# --------------------------------------------------
st.markdown("### Heatmap — TV Match vs Benchmark (Top 3 Candidates)")
top3_ids = df_rank.head(3)["employee_id"].tolist()
cands_tv = get_candidates_tv(top3_ids)
fig_ht = build_tv_match_heatmap(cands_tv, bench_tv_median, title="TV Match vs Benchmark (Capped at 100%)")
st.plotly_chart(fig_ht, use_container_width=True)

# --------------------------------------------------
# TV Gap Bars — per TGV (for Top 1)
# --------------------------------------------------
st.markdown(f"### TV Gap — Top Candidate: {top['fullname']} (per TGV)")
top1_tv = get_candidates_tv([top["employee_id"]])
top1_tv_row = top1_tv.iloc[0] if not top1_tv.empty else pd.Series()

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Competency TVs**")
    st.plotly_chart(build_tv_gap_bar(top1_tv_row, bench_tv_median, group="Competency"), use_container_width=True)
    st.markdown("**Behavior TVs**")
    st.plotly_chart(build_tv_gap_bar(top1_tv_row, bench_tv_median, group="Behavior"), use_container_width=True)
with g2:
    st.markdown("**Cognitive TV**")
    st.plotly_chart(build_tv_gap_bar(top1_tv_row, bench_tv_median, group="Cognitive"), use_container_width=True)
    st.markdown("**Readiness TVs**")
    st.plotly_chart(build_tv_gap_bar(top1_tv_row, bench_tv_median, group="Readiness"), use_container_width=True)

# --------------------------------------------------
# Strengths Intelligence
# --------------------------------------------------
st.markdown("## Strengths Intelligence")

top20_ids = df_rank.head(20)["employee_id"].tolist()
top20_tv = get_candidates_tv(top20_ids)
freq = strengths_frequency(top20_tv)
if not freq.empty:
    st.markdown("### Top Strengths among Top 20 Candidates")
    fig_strength = go.Figure(go.Bar(
        x=freq.index.tolist(), y=freq.values.tolist(),
        marker_color=THEME["accent"],
        text=freq.values.astype(int), textposition="outside"
    ))
    fig_strength.update_layout(height=550, xaxis_title="Strength Theme", yaxis_title="Frequency")
    st.plotly_chart(fig_strength, use_container_width=True)
else:
    st.write("_No strengths data available for Top 20 candidates._")

st.markdown(f"### Strength Gap — Top Candidate vs Benchmark Cohort")
if not top1_tv.empty:
    sgap = strengths_gap_for_candidate(top1_tv_row, bench_df)
    if not sgap.empty:
        fig_sgap = go.Figure(go.Bar(
            x=sgap["strength"].tolist(), y=sgap["benchmark_%"].tolist(),
            marker_color=THEME["accent"],
            text=[f"{v:.0f}%" for v in sgap["benchmark_%"].tolist()], textposition="outside"
        ))
        fig_sgap.update_layout(height=480, xaxis_title="Strength", yaxis_title="% of Benchmark Cohort with this Strength")
        st.plotly_chart(fig_sgap, use_container_width=True)
    else:
        st.write("_Top candidate has no strength data._")

# --------------------------------------------------
# Automated HR Insights (no emojis, bullet points)
# --------------------------------------------------
st.markdown("## Automated HR Insights")
top_score = float(pd.to_numeric(top.get("final_match_rate"), errors="coerce"))
insights = []

# Fit summary
if top_score >= 90:
    insights.append(f"- Best-Matched Talent shows very high overall fit ({top['final_match_rate']}%).")
elif top_score >= 80:
    insights.append(f"- Best-Matched Talent is well aligned with most role requirements.")
else:
    insights.append(f"- Best-Matched Talent will require significant development before placement.")

# Strength amplifiers
strengths_msgs = []
if safe(top.get('comp_rate')) >= 100: strengths_msgs.append("Competency")
if safe(top.get('strength_rate')) >= 100: strengths_msgs.append("Strength DNA")
if safe(top.get('cognitive_rate')) >= 100: strengths_msgs.append("Cognitive")
if safe(top.get('behavior_rate')) >= 100: strengths_msgs.append("Behavior")
if safe(top.get('readiness_rate')) >= 100: strengths_msgs.append("Readiness")
if strengths_msgs:
    insights.append("- Strength drivers: " + ", ".join(strengths_msgs))

# Risk indicators
risks_msgs = []
if safe(top.get('comp_rate')) < 85: risks_msgs.append("Competency")
if safe(top.get('behavior_rate')) < 85: risks_msgs.append("Behavior")
if safe(top.get('readiness_rate')) < 85: risks_msgs.append("Readiness")
if risks_msgs:
    insights.append("- Risks to monitor: " + ", ".join(risks_msgs))

# Development suggestions
if "Behavior" in risks_msgs:
    insights.append("- Development: stakeholder communication and collaboration coaching.")
if "Readiness" in risks_msgs:
    insights.append("- Development: increase role exposure and scope to accelerate readiness.")
if "Competency" in risks_msgs:
    insights.append("- Development: targeted technical upskilling aligned to role priorities.")

for bullet in insights:
    st.markdown(bullet)

# --------------------------------------------------
# Match Rate Distribution
# --------------------------------------------------
st.markdown("### Match Rate Distribution")
hist = go.Figure(data=[go.Histogram(
    x=df_rank["final_match_rate"], nbinsx=20, marker_color=THEME["base_soft"]
)])
hist.update_layout(height=320, xaxis_title="Final Match Rate (%)", yaxis_title="Count")
st.plotly_chart(hist, use_container_width=True)

# --------------------------------------------------
# Compare Finalists (Top 10, radar)
# --------------------------------------------------
st.markdown("### Compare Finalists (Radar)")
top10 = df_rank.head(10).copy()
pool_opts = top10["employee_id"].tolist()
sel_compare = st.multiselect(
    "Choose up to 3 finalists",
    options=pool_opts,
    max_selections=3,
    default=st.session_state.get("compare_ids", []),
    format_func=lambda x: f"{x} — {top10.loc[top10['employee_id']==x,'fullname'].values[0]}"
)
st.session_state["compare_ids"] = sel_compare

if sel_compare:
    compare_rows = top10[top10["employee_id"].isin(sel_compare)].to_dict(orient="records")
    fig_compare = plot_radar_multi(compare_rows, bench_tgv_avg)
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.caption("Tip: pick 2–3 names from the Top 10 to compare radar profiles.")

# --------------------------------------------------
# AI Job Profile & Insights (English, bullet lists)
# --------------------------------------------------
st.markdown("## AI Job Profile & Insights")
try:
    ai_text = call_ai(role_name, job_level_ui, role_purpose, df_rank.head(5))
    st.markdown(ai_text)
except Exception as e:
    st.error(f"AI call failed: {e}")
