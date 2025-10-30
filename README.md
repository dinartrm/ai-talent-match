🔗 **Live App:** [AI Talent Match Dashboard (Streamlit Deployment)](https://ai-talent-match-dinartrimurti.streamlit.app/)  
Explore the live, fully deployed version of the AI Talent Match & Storytelling Dashboard.


# AI Talent Match & Storytelling Dashboard

An intelligent HR analytics system that compares employees’ capabilities against benchmark top performers to identify best-fit candidates for specific roles. The app provides data-driven visual analytics, AI-generated job profiles, and automated HR insights to support **Talent Intelligence and Succession Planning**.

---

## Overview

This project combines **data analytics, AI narrative generation, and interactive visualization** to help HR teams:
- Evaluate employees’ readiness for specific roles.
- Identify high-potential talents and benchmark top performers.
- Generate automatic job profiles and competency insights using AI.

All processing and visualization are implemented in **Python** and **Streamlit**, with data stored in a **Supabase (PostgreSQL)** database.

---

## Key Features

### Talent Benchmarking
- Compares each employee’s **Competency**, **Strength DNA**, **Cognitive**, **Behavior**, and **Readiness** scores.
- Benchmarks are defined from **top performers** in the selected job level (Grade III–V).

### Interactive Visual Analytics
- **Radar chart** to compare top candidates vs benchmark.
- **Heatmap** and **bar charts** for TV and TGV gap analysis.
- **Strength distribution** to identify frequent high-performing traits.

### Automated HR Insights
- AI summarizes top candidates’ fit levels and improvement areas.
- Insight bullets categorized as **fit summary**, **strength amplifiers**, and **development suggestions**.

### AI Job Profile Generation
- Uses **OpenRouter API (Claude 3 Haiku)** to produce:
  - Job Requirements (bulleted list)
  - Narrative Job Description
  - Key Competencies (with tools)
  - Why Top Candidates Rank Highest

### Data-Driven Insights
The analysis notebook `discover_pattern.ipynb` explores:
- Distribution of top performers.
- Tenure, grade, and education patterns.
- Psychometric availability and data integrity.
- Feature engineering and rating cleaning steps.

---

## How It Works

1. **User inputs:**
   - Role Name (free text)
   - Job Level (Junior / Middle / Senior)
   - Role Purpose (short description)
   - Benchmark Employees (optional)

2. **System process:**
   - Fetches relevant employees from Supabase database.
   - Calculates **match rates** across 5 capability dimensions.
   - Generates ranking and visual analytics.

3. **Output:**
   - Top candidate ranking table.
   - Benchmark vs candidate radar visualization.
   - TV and TGV gap analysis.
   - Strength insights.
   - AI-generated Job Profile & Summary Report.

---

## Data Source

Dataset: `employees_scored.csv`  
Database: **PostgreSQL (Supabase)**  
Table: `employees_scored`  

Contains:
- Employee demographic info  
- Psychometric test results  
- Competency indicators  
- Strength profiling (5 dominant strengths)

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/dinartrm/ai-talent-match.git
cd ai-talent-match/talent_app
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure secrets
```toml
# Database
DB_HOST = "YOUR_SUPABASE_HOST"
DB_PORT = "5432"
DB_NAME = "YOUR_DATABASE_NAME"
DB_USER = "YOUR_DB_USER"
DB_PASS = "YOUR_DB_PASSWORD"

# OpenRouter API
OPENROUTER_API_KEY = "sk-or-yourapikey"
```
### 4. Run the app
```bash
streamlit run app.py
```
