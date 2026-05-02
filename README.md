# ⚡ Self-Healing Data Pipeline Debugger

> An agentic RAG system that monitors Apache Airflow pipelines, diagnoses failures
> using a vector knowledge base, and generates AI-powered fix recommendations — automatically.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Airflow](https://img.shields.io/badge/Airflow-2.8-017CEE?style=flat-square&logo=apacheairflow)
![dbt](https://img.shields.io/badge/dbt-1.7-FF694B?style=flat-square&logo=dbt)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.62-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?style=flat-square)

---

## What It Does

Most data pipeline failures produce cryptic logs that require an experienced engineer
to diagnose. This project automates that process end-to-end:

1. **Monitors** Airflow every 30 seconds for failed tasks
2. **Retrieves** relevant context from a vector knowledge base (past incidents, runbooks, schema docs)
3. **Diagnoses** the failure using LLaMA 3.3 70B via Groq with RAG-augmented prompts
4. **Generates** a structured report: root cause, recommended fix, confidence score, runbook reference
5. **Visualizes** everything in a live Streamlit dashboard

```
Pipeline fails → Agent detects → RAG retrieves context → LLM diagnoses → Fix recommended
     ↑                                                                           ↓
     └─────────────────── Incident stored in knowledge base ────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
│   Airflow DAGs · dbt Models · Postgres/pipeline_db              │
└──────────────┬──────────────────────────────┬───────────────────┘
               │ failure/log                  │ healthy tick
               ▼                              ▼
┌──────────────────────────┐    ┌─────────────────────────────┐
│    Monitoring Agent      │    │      Run Status Store       │
│  Polls Airflow REST API  │    │   SQLite · Postgres          │
│  Classifies error type   │    └─────────────────────────────┘
└──────────┬───────────────┘
           │ queries
           ▼
┌──────────────────────────────────────────────┐
│           RAG Knowledge Base                 │
│  Past incidents · dbt schema docs            │
│  Runbooks · SQL models                       │
│  ChromaDB + sentence-transformers            │
└──────────────────┬───────────────────────────┘
                   │ context
                   ▼
┌──────────────────────────────┐
│         LLM Brain            │
│  LLaMA 3.3 70B via Groq      │
│  ReAct agent loop            │
└──────────────┬───────────────┘
               │ fix proposal
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Layer                                  │
│   Root-cause report · SQL fix · Confidence score · Runbook ref  │
└──────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Incident Memory Store                           │
│   Learns from resolved incidents · improves future retrieval    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Failure Types Handled

| Error Type | Example | Detection Method |
|---|---|---|
| `schema_mismatch` | Upstream renamed `region` → `geo_region` | Column not found in logs + schema validation |
| `data_quality_failure` | NULL primary keys, negative amounts, orphaned FKs | dbt-style quality gate checks |
| `task_timeout` | GROUP BY on unindexed 50M-row table | AirflowTaskTimeout + slow query detection |
| `connection_failure` | Database unreachable | Connection refused in logs |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | Apache Airflow 2.8 | Industry standard pipeline scheduler |
| Transformation | dbt Core 1.7 | Schema documentation + data quality tests |
| Database | Postgres 15 | Production-grade relational store |
| Vector DB | ChromaDB | Local, persistent, no API cost |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, 384-dim |
| LLM | LLaMA 3.3 70B via Groq | Free tier, extremely fast inference |
| Agent framework | LangGraph | Stateful ReAct agent loops |
| Dashboard | Streamlit + Plotly | Rapid ML-focused UI |
| Infra | Docker Compose | Reproducible local environment |

**Total cost to run: $0** — every component is free or open source.

---

## Project Structure

```
pipeline-debugger/
├── airflow/
│   └── dags/
│       ├── dag_01_healthy.py          # Baseline working pipeline
│       ├── dag_02_schema_error.py     # Failure: column renamed upstream
│       ├── dag_03_data_quality.py     # Failure: nulls + bad data
│       └── dag_04_timeout.py          # Failure: query timeout
├── dbt_project/
│   └── models/staging/
│       ├── stg_orders.sql             # Data transformation
│       └── schema.yml                 # Column docs + quality tests
├── phase3/                            # RAG Knowledge Base
│   ├── rag/
│   │   ├── ingest.py                  # Embeds docs into ChromaDB
│   │   └── retriever.py               # Semantic search interface
│   ├── data/
│   │   ├── incidents/                 # JSON incident files
│   │   └── chroma_db/                 # Persisted vector store
│   └── docs/runbooks/                 # Markdown fix guides
├── phase4/
│   ├── agent.py                       # Monitoring agent (main)
│   └── reports/                       # AI diagnosis JSON reports
├── phase5/
│   └── dashboard.py                   # Streamlit monitoring UI
├── phase6/
│   └── eval.py                        # RAG retrieval evaluation
├── scripts/
│   ├── init_db.sql                    # Seeds Postgres
│   └── log_exporter.py               # Exports Airflow failures
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11
- Git

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/pipeline-debugger.git
cd pipeline-debugger

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start the infrastructure

```bash
docker compose up -d
# Wait ~2 minutes for first startup

# Verify all services are healthy
docker compose ps
```

### 3. Initialize the database

```bash
docker compose exec postgres psql -U airflow -c "CREATE DATABASE pipeline_db;"
docker compose exec postgres psql -U airflow -d pipeline_db -c "
  CREATE TABLE raw_orders (order_id VARCHAR(50), customer_id INTEGER,
    amount NUMERIC(10,2), status VARCHAR(20), created_at TIMESTAMP, region VARCHAR(10));
  CREATE TABLE raw_customers (customer_id INTEGER PRIMARY KEY, name VARCHAR(100),
    email VARCHAR(100), tier VARCHAR(10), created_at TIMESTAMP);
  CREATE TABLE stg_orders (order_id VARCHAR(50) PRIMARY KEY, customer_id INTEGER,
    amount NUMERIC(10,2), status VARCHAR(20), created_at TIMESTAMP,
    region VARCHAR(10), _loaded_at TIMESTAMP DEFAULT NOW());
  CREATE TABLE mart_daily_revenue (report_date DATE PRIMARY KEY, region VARCHAR(10),
    total_revenue NUMERIC(12,2), order_count INTEGER, _updated_at TIMESTAMP DEFAULT NOW());
"
```

### 4. Add Airflow connection

Open http://localhost:8080 (admin/admin) → Admin → Connections → +

| Field | Value |
|---|---|
| Connection Id | `pipeline_db` |
| Connection Type | `Postgres` |
| Host | `postgres` |
| Database | `pipeline_db` |
| Login | `airflow` |
| Password | `airflow` |
| Port | `5432` |

### 5. Build the RAG knowledge base

```bash
pip install chromadb sentence-transformers pyyaml
python phase3/rag/ingest.py --base-dir phase3
```

### 6. Set up Groq API key (free)

Get a free key at https://console.groq.com

```bash
export GROQ_API_KEY="gsk_your_key_here"
# Windows: $env:GROQ_API_KEY = "gsk_your_key_here"
```

### 7. Trigger failures and run the agent

In Airflow UI, trigger `dag_02_schema_mismatch`, `dag_03_data_quality_failure`,
and `dag_04_timeout_failure`. Then:

```bash
python phase4/agent.py --once
```

### 8. Launch the dashboard

```bash
pip install streamlit plotly
streamlit run phase5/dashboard.py
# Open http://localhost:8501
```

---

## How the RAG Pipeline Works

```
Error log: "column geo_region does not exist in raw_orders"
                    ↓
          Embed with sentence-transformers
                    ↓
         Search ChromaDB (cosine similarity)
                    ↓
    Top results:
      [0.764] schema_mismatch.md → "Fix: SELECT geo_region AS region"
      [0.609] schema_mismatch.md → "Known incident: platform renamed region"
      [0.526] dag_02 incident    → "Same error on 2024-03-15"
                    ↓
    Inject as context into LLM prompt
                    ↓
    LLM diagnosis:
      root_cause:  "Upstream column renamed region → geo_region"
      fix:         "UPDATE SELECT to use 'region' or alias 'geo_region AS region'"
      confidence:  "high"
      runbook:     "schema_mismatch.md"
```

---

## Evaluation Results

Run `python phase6/eval.py` to see retrieval quality metrics:

| Query | Top Result | Relevance |
|---|---|---|
| schema mismatch column renamed | schema_mismatch.md | 0.764 |
| null order_id data quality | data_quality.md | 0.658 |
| airflow task timeout slow query | timeout.md | 0.745 |
| raw_orders columns schema | schema.yml | 0.441 |
| how to fix schema mismatch | schema_mismatch.md | 0.683 |

**Mean relevance score: 0.658** — well above the 0.50 threshold for useful retrieval.

---

## Key Design Decisions

**Why separate ChromaDB collections per document type?**
Filtering by source type reduces noise. When diagnosing a timeout, schema docs
are irrelevant — filtering to `["runbooks", "incidents"]` improves precision.

**Why overlapping chunks?**
Key sentences at chunk boundaries would be split and lose context. 100-word
overlap ensures every sentence appears fully in at least one chunk.

**Why rule-based pre-classification before the LLM?**
The rule-based classifier runs in microseconds and gives the LLM a strong
prior, reducing hallucination risk on ambiguous errors.

**Why Groq over OpenAI?**
LLaMA 3.3 70B on Groq is free, extremely fast (~0.3s response), and performs
comparably to GPT-4o on structured diagnosis tasks. Total project cost: $0.

---

## What I'd Add With More Time

- **Auto-apply fixes**: When confidence > 0.90, automatically create the index
  or restart the DAG rather than just recommending it
- **Slack/PagerDuty integration**: Post diagnosis reports to Slack channels
- **dbt test integration**: Trigger dbt tests on failure and ingest results
- **Multi-pipeline support**: Monitor multiple Airflow instances simultaneously
- **Fine-tuned classifier**: Replace rule-based pre-classification with a small
  trained model on the incident corpus

---

## About

Built as a portfolio project demonstrating agentic RAG systems for data engineering.
Covers the full ML/AI engineer stack: pipeline orchestration, data quality,
vector databases, LLM agents, and production UI.