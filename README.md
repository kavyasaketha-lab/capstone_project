# Zepto Data Engineering, Analytics & GenAI Project

## Project Overview

This project implements an end-to-end data and AI workflow for **Zepto**, covering three major areas:

1. **Data Pipeline** - Web scraping, data cleaning, currency conversion, relational database design, and SQL/Pandas analysis.
2. **Analytics Pipeline** - Exploratory data analysis, data cleaning, visualization, classification modeling, model evaluation, hyperparameter tuning, and regression.
3. **Support Assistant** - A Retrieval-Augmented Generation (RAG) application using embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

The project demonstrates a complete workflow from **raw data collection → data processing → storage → analytics → machine learning → GenAI application development**.

---

# Repository Structure

```text
capstone_project/
│
├── data_pipeline/
│   
│   ├── pipeline.py
│   ├── Output/
│   ├── zepto_books.db
│   ├── requirements.txt
│   └── README.md
│
├── analytics/
│   ├── EDA.py
│   ├── MLModelling.py
│   ├── titanic.csv
│   ├── PlotCharts/
│   ├── ModelCharts/
│   ├── requirements.txt
│   └── README.md
│
├── support_assistant/
│   ├── Documents/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── chromadbcretion.py
│   ├── Langraph.py
│   ├── fastAPI.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── requirements.txt
└── README.md
```

> The exact filenames may vary slightly depending on the implementation. Each module contains its own detailed README documenting its implementation and validation.

---

#  Project Architecture

The overall project consists of three independent but complementary pipelines.

```text
                       ZEpto Project
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
      Data Pipeline     Analytics       Support Assistant
             │           Pipeline             │
             │              │                 │
             ▼              ▼                 ▼
      Web Scraping      Titanic Data      Policy Documents
             │              │                 │
             ▼              ▼                 ▼
        Cleaning          EDA             Chunking
             │              │                 │
             ▼              ▼                 ▼
       Currency         ML Models         Embeddings
       Conversion           │                 │
             │              ▼                 ▼
             ▼         Evaluation         ChromaDB
         SQLite             │                 │
             │              ▼                 ▼
             ▼         Model Pipeline      LangGraph
      SQL + Pandas            │                 │
                            ▼                 ▼
                         Predictions       FastAPI
```

---

## Module Documentation

For implementation details, execution instructions, outputs, design decisions, and module-specific validation:

* `data_pipeline/README.md`
* `analytics/README.md`
* `support_assistant/README.md`

The root README provides the **overall project architecture and navigation**, while each module README contains the detailed evidence required for that module.