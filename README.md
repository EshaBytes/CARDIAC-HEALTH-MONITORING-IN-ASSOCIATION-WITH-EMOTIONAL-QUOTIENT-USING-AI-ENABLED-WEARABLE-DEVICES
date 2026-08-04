
# CardioEQ AI — Explainable, Unsupervised Cardiovascular Intelligence Platform

🔗 **Live demo:** [cardioeq-ai.onrender.com](https://cardioeq-ai.onrender.com)

An AI-powered cardiovascular health monitoring and analytics platform built around a real wearable sensor cohort: **20 subjects × 4 activities** (sitting, walking, running, cognitive task), captured via PPG, GSR, a 3-axis accelerometer/gyroscope (MPU-6050), skin temperature (DS18B20), a medical-grade pulse oximeter, and environmental temperature/humidity.

There are no clinician labels in this dataset — no subject is classified "healthy" or "at risk" by a doctor. That's a common constraint in early-stage health-tech research, and exactly the scenario unsupervised learning fits: the model learns what normal physiological behavior looks like directly from the cohort's own data, then flags deviations from that learned norm as elevated risk.

The system consists of three modules:

- **Backend API** — FastAPI services, JWT auth, data processing, PDF report generation, ML inference.
- **Frontend Dashboard** — React web app for visualization, subject management, longitudinal monitoring, research analytics.
- **ML Pipeline** — signal preprocessing, feature extraction, unsupervised cardiovascular risk assessment, explainability, dataset generation.

This isn't a mockup — the ML pipeline runs on real sensor data, the backend is a tested FastAPI service (`smoke_test.py` drives the real routers/services/ml_core through an in-memory Mongo), and the frontend is wired end-to-end. See **"What's real vs. what you still need to do"** below before treating this as production-ready.

---


## Architecture

```
raw_data/ (PPG/GSR/IMU/temp CSVs, 20 subjects × 4 activities, ~2.86M rows)
        │
        ▼
┌────────────────────────┐
│ ml-pipeline/           │  offline, run once (or when re-training)
│ build_dataset.py       │  → feature_extraction (30s windows, HRV,
│                        │    stress, motion) → activity normalization
│                        │    → RobustScaler → unsupervised_risk.py
│                        │    (GMM + Isolation Forest, fused 60/40,
│                        │    percentile-ranked 0–100)
│                        │  → Heart Health Score (additive, explainable)
│                        │  → rule-based insight generation
└──────────┬─────────────┘
           │ writes data_processed/*.json + persists model artifacts
           ▼
┌───────────────────────────┐        ┌──────────────────────────┐
│ backend/ (FastAPI)        │◄──────►│ MongoDB                  │
│ - JWT auth                │        │ subjects, sessions,      │
│ - subjects/sessions       │        │ timeseries_features,     │
│ - explainability API      │        │ insights,                │
│ - population API          │        │ population_stats, users  │
│ - EQ research/correlation │        └──────────────────────────┘
│ - AI health assistant     │
│ - PDF reports             │
│ - live upload → inference │
└──────────┬────────────────┘
           │ REST API (JSON)
           ▼
┌────────────────────────┐
│ frontend/ (React)      │  Landing, Sign Up/In, Forgot Password, Profile,
│ Vite + Tailwind        │  Dashboard (Time Series / Explainability /
│ + Recharts             │  Population / Longitudinal / Insights),
└────────────────────────┘  EQ Research page, AI assistant widget, PDF download
```

---

## Features

- Cardiovascular risk assessment via Gaussian Mixture Model (GMM) + Isolation Forest, fused into a 0–100 score
- Explainable, distance-based risk scoring with per-feature deviation breakdowns
- Additive Heart Health Score with per-biomarker breakdown
- Wearable biosensor recording upload, validation, and preprocessing
- Interactive ECG / physiological time-series visualization
- Longitudinal cardiovascular monitoring across sessions
- Population-level analytics and cohort/similar-profile benchmarking
- Rule-based explainable insight generation (pattern → why → impact → recommendation)
- EQ (Emotional Quotient) questionnaire, EQ profile visualization, and EQ research dashboard
- Correlation analysis between EQ and physiological biomarkers
- AI-powered health assistant (template mode by default; narrates precomputed facts via an LLM if configured — never invents numbers)
- PDF cardiovascular health report generation
- JWT authentication, role-based access control (Admin/User)
- Automatic recomputation of population statistics after every upload

---

## Technology Stack

**Backend:** FastAPI, Python 3.11, MongoDB (Motor & PyMongo), JWT auth, Passlib, scikit-learn, ReportLab, Pydantic

**Frontend:** React 18, Vite, React Router, Tailwind CSS, Recharts, Axios, Vitest

**Machine Learning:** NumPy, Pandas, scikit-learn, Gaussian Mixture Model (GMM), Isolation Forest

---

## Wearable Biosensor Platform

| Sensor | Measurement |
|---|---|
| HRM-2511E PPG | Heart rate & pulse waveform |
| GSR sensor | Skin conductance / stress response |
| MPU-6050 IMU | Motion & activity recognition |
| DS18B20 | Skin temperature |
| Medical-grade pulse oximeter | Blood oxygen saturation (SpO₂) |

---

## Dataset

20 healthy participants (12 male, 8 female, aged 19–26, BMI 17–43), each recorded across 4 activities — sitting, walking, running, seated cognitive task — sampled at ~80–150 Hz.

- 20 participants, 80 recording sessions
- ~2.86 million raw sensor observations
- 25 variables per file: PPG amplitude, GSR conductance, 3-axis accel/gyro, skin temperature, SpO₂, demographics, environmental readings
- 30-second non-overlapping analysis windows
- 7 primary physiological biomarkers extracted per window: Heart Rate, RR Interval, RMSSD, SDNN, Stress Index, Recovery Rate, Motion Intensity (age/BMI computed but excluded from training — used only for cohort benchmarking)

---

## Machine Learning Pipeline

1. Signal preprocessing & noise filtering
2. Beat detection & RR interval extraction
3. Heart Rate Variability (HRV) computation
4. Activity-wise normalization (z-scored within activity group, so the model learns "how risky," not "which activity")
5. RobustScaler feature scaling (median/IQR-based — robust to motion artifacts)
6. Window-level feature extraction
7. Unsupervised cardiovascular risk estimation
8. Explainability generation

### Unsupervised Cardiovascular Risk Assessment

Fully unsupervised anomaly detection — no labeled clinical data required. GMM (density-based) and Isolation Forest (partition-based) independently score every window; scores are fused 60/40 and percentile-ranked to a 0–100 risk score. On this cohort the two models agree at Pearson r = 0.941, the core evidence the fused score isn't an artifact of either algorithm alone.

**Honesty about the dataset's limits:**
- PPG-based beat detection (adaptive band-pass filtering + threshold crossing) yields different absolute HRV magnitudes than a clinically-tuned ECG R-peak detector. `scoring.py` derives "healthy" reference ranges from this cohort's own low-anomaly windows where enough data exists, falling back to generic clinical-literature defaults otherwise.
- With no clinician labels anywhere, every risk category (Healthy / Mild Risk / Moderate Risk) is **relative to this cohort** — "more/less anomalous than the other 19 subjects" — not a clinical diagnosis. Surfaced explicitly in the UI and every report.
- A real Pan–Tompkins QRS detector exists in `feature_extraction.py` for future raw-waveform ingestion, but is currently dormant — this dataset ships precomputed beat flags upstream.
- EQ is not something a PPG/GSR wearable measures, so it's collected separately via a short, original 15-item questionnaire (not a licensed instrument like Bar-On EQ-i). Only 6 of 20 subjects have completed it so far — EQ correlation results are exploratory, not statistically confirmed. `composure_index_proxy` and `cognitive_load_index` are physiological proxies derived from stress/recovery/HRV dynamics, clearly labeled as proxies and never presented as validated EQ measurements.

---

## Explainability

Since the risk model is fully unsupervised, feature-attribution methods like SHAP or LIME don't directly apply. Instead, CardioEQ AI uses a **distance-based explainability strategy** that:

- Identifies the nearest healthy Gaussian cluster
- Computes normalized deviations for physiological biomarkers
- Selects the most influential features
- Generates human-readable explanations for why a subject received a particular risk score

---

## Emotional Quotient (EQ) Research Module

- Digital EQ questionnaire, automatic EQ score computation, EQ profile visualization
- Interactive research dashboard correlating: EQ Score, Cardiovascular Risk Score, HRV (RMSSD), Stress Index, Recovery Rate, Resting Heart Rate, Composure Index Proxy

---

## Repository Structure

```
backend/
├── app/
│   ├── routers/       auth, subjects, population, reports, research, assistant
│   ├── services/       inference, retrain, report_pdf, assistant_service, population_recompute
│   ├── ml_core/         feature_extraction, risk_model, unsupervised_risk, scoring,
│   │                    fingerprint, insights, eq_questionnaire
│   ├── models/            Pydantic schemas
│   ├── main.py, db.py, security.py, config.py
├── scripts/           seed_mongo.py, link_existing_users.py
└── smoke_test.py

frontend/
└── src/
    ├── pages/         Landing, SignIn/Up, ForgotPassword, Profile, Settings,
    │                  SubjectDashboard, SubjectsOverview, Research,
    │                  AdminUserManagement, AdminEqManagement, UploadRecording
    ├── components/     Sidebar, Topbar, SessionsPanel, TimeSeriesPanel,
    │                  PopulationPanel, LongitudinalPanel, ScoreRing, RiskBadge,
    │                  EcgLine, ExplainabilityPanel, AssistantWidget
    ├── context/        AuthContext, ThemeContext, ToastContext
    └── lib/            api, biomarkerDirections, format, subjectId, syncBus

ml-pipeline/
├── raw_data/           subject_metadata.csv + modified CSVs
├── data_processed/      subjects, sessions, timeseries_features, insights,
│                       population_stats, risk_model_report, unsupervised_model_report
└── build_dataset.py    single source of truth — output schema matches backend Mongo collections
```

---

## Setup

### 0. Prerequisites
Python 3.11+, Node 18+, a MongoDB instance (Atlas or self-hosted).

### 1. ML pipeline — run once, to fit the models and process your cohort
```bash
cd ml-pipeline
pip install -r ../backend/requirements.txt   # shares deps with the backend

export RAW_DATA_DIR=/path/to/your/raw_data
python build_dataset.py
```
Writes `ml-pipeline/data_processed/*.json` (ready-to-import Mongo documents) and persists fitted GMM/Isolation Forest artifacts to `backend/app/ml_core/artifacts/`, used later to score newly-uploaded subjects live via the API. Fitting runs single-threaded (`threadpool_limits`) for reproducible scores across runs.

### 2. Configure and seed MongoDB
```bash
cd backend
cp .env.example .env
```
Set `MONGODB_URI`, `MONGODB_DB_NAME`, and a real `JWT_SECRET` (e.g. `openssl rand -hex 32`) in `.env`. Optional keys: `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` (AI assistant — falls back to template-mode responses when unset), `AIR_QUALITY_API_KEY` (environmental correlation, not yet wired up).

> **Security:** rotate any credentials ever shared in a chat, ticket, or commit before using them beyond local testing.

```bash
python scripts/seed_mongo.py
```
Resetting later — `seed_mongo.py` also handles wipe + reseed at any point. By default it clears only the 5 cohort-data collections (subjects, sessions, timeseries, insights, population stats) and leaves logins untouched:
```bash
python scripts/seed_mongo.py --dry-run       # preview only, deletes nothing
python scripts/seed_mongo.py --full-wipe     # also clears users + password_resets (asks for confirmation)
python scripts/seed_mongo.py --full-wipe --yes   # skip the interactive confirmation prompt
```

### 3. Run the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Interactive API docs: `http://localhost:8000/docs`. Run `python smoke_test.py` any time to sanity-check the full API against an in-memory Mongo — no live DB connection needed.

### 4. Run the frontend
```bash
cd frontend
cp .env.example .env   # set VITE_API_BASE_URL if backend isn't on localhost:8000
npm install
npm run dev
```
Visit `http://localhost:5173` — sign up, then walk through cohort overview, upload a recording, and a subject's Time Series → Explainability → Population → Longitudinal → Insights tabs.

---

## Testing

```bash
# backend — real routers/services/ml_core against an in-memory Mongo, no live DB needed
cd backend && python smoke_test.py

# frontend
cd frontend && npm run test
```

---
 
## Screenshots
 
### Landing Page
 
<img width="1920" height="1080" alt="Screenshot (117)" src="https://github.com/user-attachments/assets/ae3f70bb-9a58-41dc-a8a2-34077b4b2447" />
Landing page showing the project overview, key features, and call-to-action.

### Profile
<img width="1920" height="1080" alt="Screenshot (146)" src="https://github.com/user-attachments/assets/97cd470c-5ef9-4be1-8a2e-9766d9acc225" />
Profile page displaying account information, physiological details, and automatically calculated BMI.

 
### Cohort Overview
 
<img width="1920" height="1080" alt="Screenshot (149)" src="https://github.com/user-attachments/assets/d5994fb2-a86c-4697-8132-35265a4ea4f7" />
Shows the subject's cardiovascular risk score, percentile, and risk category.

### Admin Cohort Overview
<img width="1920" height="1080" alt="Screenshot (147)" src="https://github.com/user-attachments/assets/1eef7003-4f4b-4fb9-8ce5-8db8677d7ae4" />
Displays all subjects along with the cohort-wide risk distribution.
 
### Subject Dashboard
<img width="1920" height="1080" alt="Screenshot (123)" src="https://github.com/user-attachments/assets/ca731aa4-2e05-47aa-889a-dc31023a8eb0" />
EQ score card and overall session summary.

### Sessions Dashboard
<img width="1920" height="1080" alt="Screenshot (125)" src="https://github.com/user-attachments/assets/e20e9bf1-c6e0-407d-b2d8-4c240d71ac33" /> 
Per-activity heart rate and RMSSD time-series visualizations.

<img width="1920" height="1080" alt="Screenshot (130)" src="https://github.com/user-attachments/assets/2f3f6d0f-876e-4cc3-9970-616f35c2cccb" />
Time-series plots of SpO₂ and skin temperature.
<br>
<img width="1920" height="1080" alt="Screenshot (129)" src="https://github.com/user-attachments/assets/c867700b-2328-4ab8-9524-4c13a98bfb62" />
Stress Index and Recovery Rate over time.
<br>
<img width="1920" height="1080" alt="Screenshot (127)" src="https://github.com/user-attachments/assets/8d520f1b-c2dc-4b6b-abf9-6b058e6d6947" />
Heart rate variability (SDNN) and RR Interval plots.

### Explainability
<img width="1920" height="1080" alt="Screenshot (131)" src="https://github.com/user-attachments/assets/a75c602f-c152-4df6-8cd3-e680f38c5a90" />
Comparison of each biomarker with its healthy reference range.
<br>
<img width="1920" height="1080" alt="Screenshot (132)" src="https://github.com/user-attachments/assets/411908c5-ced5-4874-9fae-2bca165b128d" />
Biomarker deviation analysis and cardiovascular risk explanation.

### Longitudinal Analysis
<img width="1920" height="1080" alt="Screenshot (133)" src="https://github.com/user-attachments/assets/5c8e4f5c-5836-489d-82c3-fb1617902429" />
ML risk score trends across multiple recording sessions.


 
### Upload & EQ Research
 <img width="1920" height="1080" alt="Screenshot (134)" src="https://github.com/user-attachments/assets/36e69cdf-6cec-437c-a534-4e666c6eb128" />
Upload page with the EQ self-assessment questionnaire.

### EQ Graphs
<img width="1920" height="1080" alt="Screenshot (137)" src="https://github.com/user-attachments/assets/8bac9487-f910-4bc7-b169-daf21eb67692" />
Relationship between Emotional Quotient and ML Risk Score, and between EQ and Composure Proxy.
<br>
<img width="1920" height="1080" alt="Screenshot (138)" src="https://github.com/user-attachments/assets/6a60de61-565d-4959-8740-521362839382" />
Comparison of EQ with Cognitive Load Index and RMSSD.
<br>
<img width="1920" height="1080" alt="Screenshot (139)" src="https://github.com/user-attachments/assets/6277ee6d-8ab0-41a4-9a85-e975255dc6fe" />
Correlation of EQ with Stress Index and Recovery Rate.
<br>
<img width="1920" height="1080" alt="Screenshot (140)" src="https://github.com/user-attachments/assets/aac2489a-d5b5-4871-ac79-7a100f61d45c" />
Relationship between EQ and resting heart rate.

### Reference Ranges
<img width="1920" height="1080" alt="Screenshot (141)" src="https://github.com/user-attachments/assets/352a3cac-ae79-4521-9cfa-8d013694b118" />
Activity-specific physiological reference ranges.

### Correlation Matrix
<img width="1920" height="1080" alt="Screenshot (142)" src="https://github.com/user-attachments/assets/76ebfd3e-2136-4f93-82d6-bed282dabc78" />
Correlation matrix between EQ score and cardiovascular metrics.

### Conclusion Summary
<img width="1920" height="1080" alt="Screenshot (143)" src="https://github.com/user-attachments/assets/01eb696e-7843-464d-8233-26ae4fcf86ad" />
Automatically generated research summary based on the analyzed data.

---

## What's real vs. what you still need to do

**Fully working, tested end-to-end** (`smoke_test.py` exercises all of this against a real in-memory Mongo, including the live upload pipeline): JWT auth, subject CRUD, windowed HRV/biomarker time series, the GMM + Isolation Forest risk model with per-feature GMM-native explanations, the additive Heart Health Score, population percentile benchmarking, longitudinal session comparison, rule-based insight generation, the EQ correlation research module, the AI assistant (template mode by default, LLM-narrated if a key is configured — narrates precomputed facts only, never invents numbers), PDF report generation, and live CSV upload → feature extraction → scoring.

**Stubbed — needs real infrastructure before production:**
- **Password reset emails:** forgot-password returns the reset token directly in the API response (no email service wired up). Wire up SES/Postmark/etc. before shipping.
- **Air quality / pollution data:** schema field (`air_quality_index`) exists, not populated. Wire up an air-quality API keyed by location.
- **EQ score coverage:** only 6 of 20 subjects completed the questionnaire so far — correlation results scale in reliability as more subjects complete it, no pipeline changes needed.
- **Model retraining cadence:** `build_dataset.py` is run manually. For production, schedule it (cron/Airflow) as more subjects are added, and version the artifacts.
- **Multi-tenancy / RBAC:** every authenticated user currently sees the full shared cohort. Add an `owner_org_id` filter in `backend/app/routers/subjects.py` for per-organization isolation.
- **Real-time streaming:** current implementation is offline/batch only — the Arduino streams to a phone for recording, not live to the dashboard.
- **Post-deploy migration:** `link_existing_users.py` needs one manual run against the live Atlas DB after deploying, to link existing users.

---

## Future Work

- Real-time wearable data streaming
- Mobile application support
- Clinical validation on larger, more diverse cohorts
- Integration with additional wearable sensors
- Cloud deployment with continuous monitoring
- Personalized cardiovascular health recommendations


