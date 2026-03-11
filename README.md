# MaaS Intelligent Operations Agent

End-to-end demand forecasting system with an LLM-based operations assistant.

This project is a proof-of-concept (POC) designed for the Mobility as a Service (MaaS) industry.  
It analyzes large-scale public taxi trip data from New York City to build a machine learning demand forecasting model and integrates it with an LLM-based agent workflow.

The goal is to enable operations personnel to query data and predicted vehicle demand using natural language instead of writing SQL or analytical scripts.

## Business Value

### Data-driven operations
Raw trip data is transformed into actionable insights that can assist dispatch systems in allocating vehicles to high-demand areas.

### Lower technical barriers
A large language model (LLM) interface allows operations managers to retrieve demand predictions without writing SQL or interacting directly with machine learning models.

### Scalability
The prototype is implemented locally with modular components.  
The data pipeline design is compatible with distributed processing environments such as Databricks and PySparkSQL.

## System Architecture

The project implements an end-to-end workflow consisting of three main components.

### Data Engineering

Libraries: `pandas`, `pyarrow`

- Processes large-scale NYC TLC taxi trip data in Parquet format.
- Applies domain-specific filtering rules such as invalid fare values and inconsistent timestamps.
- Data retention after cleaning is approximately 96 percent.

### Data Science

Libraries: `scikit-learn`, `XGBoost`

- Feature engineering includes temporal variables such as hour of day and day of week.
- Additional features include rush-hour indicators and weekend flags.
- An XGBoost regression model is trained to estimate hourly taxi demand per zone.
- Model performance: Mean Absolute Error (MAE) approximately 11.64.

### AI Application

Libraries: `LangGraph`, `Groq API`

- Implements a ReAct-style agent using the LangGraph framework.
- Enables tool calling so the language model can extract parameters from user queries and call the demand prediction function automatically.

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt

## Configure Environment Variables

A Groq API key is required to run the LLM component.

```bash
export GROQ_API_KEY="your_api_key_here"
```

---

## Run the Pipeline

The project is organized as modular scripts executed sequentially.

```bash
# Phase 1: Data preprocessing and feature engineering
python data_preprocessing.py

# Phase 2: Train demand prediction model
python model_training.py

# Phase 3: Start the MaaS operations agent
python maas_agent.py
```

---

## Example Usage

### User Query

```text
Estimate how many vehicles will be required at location 1 on Friday at 7 PM.
```

### Agent Workflow

1. The agent determines that the `predict_taxi_demand` tool should be used.

2. Parameters are extracted automatically:

```json
{
  "location_id": 1,
  "hour": 19,
  "day_of_week": 4,
  "is_weekend": 0,
  "is_rush_hour": 1
}
```

3. The local XGBoost model generates a prediction.

### Example Response

```text
The predicted vehicle demand for location 1 at 19:00 on Friday is approximately 6 vehicles.
```