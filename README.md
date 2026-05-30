# 5 Guys — H&M Customer Intelligence Pipeline

**Team:** Riya · Atharva · Farhan · Sacharith · Shubhendu  
**Dataset:** H&M Kaggle — 1.37M customers · 9.1M transactions · 105K products

---

## Business Question

> *Which customers are worth targeting — and what do we recommend to them?*

**Answer 1 — Customer Value Classifier:** Predicts High-Value vs Low-Value per customer using graph structure (PageRank + Community) on top of tabular degree features.

**Answer 2 — Semantic Recommendation Engine:** A store manager types a plain-English product description and receives the 5 most similar products by meaning, filtered by section with colour variants shown.

The classifier tells you **WHO** to contact. The semantic search tells you **WHAT** to recommend. Both are powered by the same graph.

---

## Key Results

| Metric | Value |
|---|---|
| Classifier F1 (macro) | **0.8665** |
| Accuracy lift (baseline → enriched) | **81.02% → 86.65% (+5.62pp)** |
| High-Value Precision lift | **0.73 → 0.87 (+14pp)** |
| PageRank hub gap | **137× (max 25.34 / median 0.185)** |
| Community coverage (top 5) | **43% of all customers** |
| Top semantic search score | **0.89** |
| Merge success rate | **100%** |

---

## Architecture

```
Raw CSV (Kaggle)
      ↓  DuckDB — clean, deduplicate, export parquet
Neo4j Graph — Customer · Article · PURCHASED · CO_PURCHASED
      ↓  Neo4j GDS — PageRank + Louvain written to Customer nodes
Feature Matrix — tabular degree counts + graph-derived columns
      ↓  RandomForest classifier (scikit-learn)
Customer Value Prediction — High-Value / Low-Value

Parallel track:
Article descriptions → all-MiniLM-L6-v2 → Qdrant vector store
      ↓
Streamlit Dashboard — product search + customer loyalty
```

Five tools, each chosen for its scale: DuckDB cleans, Neo4j connects, GDS scores, scikit-learn classifies, Qdrant retrieves, Streamlit ships.

---

## Repository Structure

```
├── notebooks/
│   ├── 02_graph_load.ipynb        # Neo4j LOAD CSV + MERGE, schema creation
│   ├── 03_graph_analytics.ipynb   # GDS PageRank + Louvain, enriched ML
│   ├── 04_ml.ipynb                # Baseline ML pipeline
│   └── 05_embeddings.ipynb        # Qdrant embedding pipeline
├── session 7/
│   └── app.py                     # Streamlit two-tab dashboard
├── docs/
│   └── ml_and_similarity_design.md  # Design document
├── report/
│   └── scale_up_reasoning.md      # Scale-up reasoning
├── models/
│   ├── customer_value_classifier.pkl      # Baseline model
│   ├── customer_value_classifier_s6.pkl   # Enriched model
│   └── s5_matrix.parquet                  # Feature matrix with node_id index
├── data/
│   └── parquet/
│       └── articles.parquet
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Docker Desktop
- Neo4j Desktop with Graph Data Science (GDS) plugin installed
- Anaconda or virtualenv

### Install dependencies

```bash
pip install qdrant-client sentence-transformers pandas duckdb \
            neo4j scikit-learn joblib streamlit matplotlib seaborn
```

### Start services

```bash
# Qdrant vector database
docker run -p 6333:6333 \
  -v "C:/Jupyter/abc/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant

# Neo4j — start from Neo4j Desktop on port 7688
# Ensure GDS plugin is installed via the Plugins tab
```

### Environment variables

```bash
# Windows CMD
set NEO4J_PASSWORD=your_password

# PowerShell
$env:NEO4J_PASSWORD="your_password"
```

---

## How to Run

Run the notebooks in this order:

| Step | Notebook | What it does |
|---|---|---|
| 1 | `02_graph_load.ipynb` | Loads CSV data into Neo4j, creates constraints and relationships |
| 2 | `04_ml.ipynb` | Builds baseline ML pipeline, saves model + feature matrix |
| 3 | `05_embeddings.ipynb` | Embeds article descriptions into Qdrant |
| 4 | `03_graph_analytics.ipynb` | Runs GDS algorithms, enriches ML pipeline, reports before/after |

Then launch the dashboard:

```bash
cd "C:/Jupyter/abc/session 7"
streamlit run app.py
```

---

## Graph Schema

### Node Labels

| Label | Key Property | Derived Properties |
|---|---|---|
| `Customer` | `customerId` | `age`, `ageBand`, `memberStatus`, `pagerank`*, `community`* |
| `Article` | `articleId` | `name`, `productType`, `garmentGroup`, `storeSection`, `detail_desc` |
| `StoreSection` | `storeSectionId` | `name` |
| `Department` | `departmentId` | `name`, `indexGroup` |
| `ProductGroup` | `productGroupId` | `name` |

*Written by Neo4j GDS during graph analytics.

### Relationship Types

| Type | From → To | Key Properties |
|---|---|---|
| `PURCHASED` | Customer → Article | `txDate`, `price`, `yearMonth` |
| `CO_PURCHASED` | Article → Article | `timesBoughtTogether`, `supportScore` |
| `IN_SECTION` | Article → StoreSection | — |
| `BELONGS_TO_DEPT` | Article → Department | — |
| `IN_GROUP` | Article → ProductGroup | — |

109,684 `CO_PURCHASED` edges are computed from transaction pairs — not present in the raw data.

### GDS Projection

```cypher
CALL gds.graph.project(
  'customer_graph',
  ['Customer', 'Article'],
  {PURCHASED: {orientation: 'UNDIRECTED'}}
)
```

---

## Machine Learning

### Target Variable

`customer_value` — `High-Value` if 2020 spend > median, `Low-Value` otherwise.
- High-Value: 15.3% (210,524 customers)
- Low-Value: 84.7% (1,161,456 customers)

### Tabular Features

| Feature | Source | Type |
|---|---|---|
| `purchase_degree` | Cypher COUNT all-time | Numerical |
| `purchase_degree_2020` | Cypher COUNT 2020 only | Numerical |
| `sections_visited` | Cypher COUNT DISTINCT sections | Numerical |
| `age` | Neo4j node property | Numerical |
| `ageBand` | Neo4j node property | Categorical |
| `memberStatus` | Neo4j node property | Categorical |

### Graph-Derived Features

| Feature | Source | Type |
|---|---|---|
| `pagerank` | `gds.pageRank.write` (dampingFactor=0.85) | Numerical |
| `community` | `gds.louvain.write` (maxLevels=10) | Categorical (cast to str) |

### Leakage Prevention

- `train_test_split(stratify=y)` runs **before** any preprocessing
- `StandardScaler` and `OneHotEncoder` live **inside** the `sklearn.Pipeline`
- The scaler only fits on `X_train` — never sees test rows

### Baseline vs Enriched Results

| Metric | Baseline (degree only) | Enriched (+pagerank +community) | Delta |
|---|---|---|---|
| Accuracy | 81.02% | 86.65% | +5.62pp |
| F1 (macro) | 0.8049 | 0.8665 | +0.0616 |
| High-Value Precision | 0.73 | 0.87 | +14pp |
| High-Value Recall | 0.98 | 0.87 | −11pp |

The recall drop is expected — the enriched model became more selective by requiring both the right spending pattern and the right structural position. For a targeted loyalty campaign, precision (0.87) matters more than recall.

---

## Semantic Search

**Model:** `all-MiniLM-L6-v2` — 384-dimensional, cosine distance, CPU-friendly  
**Field embedded:** `detail_desc` — whole-field embedding, no chunking needed  
**Collection:** `hm_articles` in Qdrant — 43,404 unique description vectors  
**Deduplication:** identical descriptions produce one vector; all `article_ids` stored in payload

```python
results = client.query_points(
    collection_name="hm_articles",
    query=model.encode("floral jersey dress").tolist(),
    limit=5,
    with_payload=True
).points
```

Top search score achieved: **0.89** cosine similarity.

---

## Dashboard

```bash
cd "C:/Jupyter/abc/session 7"
streamlit run app.py
```

### Tab 1 — Product Search
- Type a natural language product description
- Returns top-N results ranked by cosine similarity
- Shows product name, description, colour variants, section, and local image

### Tab 2 — Customer Loyalty
- **Individual lookup:** paste `customerId` → pulls features from Neo4j → predicts High-Value / Low-Value with confidence scores
- **Top 10 by PageRank:** most structurally central buyers in the purchase network
- **Community breakdown:** top 5 purchasing persona clusters by size

---

## Graph Analytics Findings

### PageRank
- Max: 25.34 · Median: 0.185 · **Hub gap: 137×**
- 5 of the top 10 customers are in their 20s — Young Fashion drives the network core
- Two 60+ customers appear in the top 10 — unexpected older hubs worth targeting

### Louvain Community Detection
- 667K communities found · Modularity: 0.260 (real structure confirmed)
- Top 5 communities cover **43% of all customers**
- Largest community: 179,558 customers (13.1%) — Young Fashion Core Buyers (46% in their 20s)

---

## Reproducibility

The full pipeline is reproducible from scratch:

```bash
docker compose down -v   # wipe everything
docker compose up -d     # restart services
# re-run notebooks in order: 02 → 04 → 05 → 03
# node and relationship counts match every single time
```

---

## Scale-Up Reasoning

| Tool | Proof-of-Concept | Production | Threshold |
|---|---|---|---|
| ETL | DuckDB | Apache Spark | ~100 GB |
| Graph DB | Neo4j Community | Neo4j Enterprise | >10B relationships |
| ML | scikit-learn | Spark MLlib | ~100 GB |
| Vectors | Qdrant Docker | Qdrant Cloud | ~few million vectors |
| Dashboard | Streamlit local | Streamlit Cloud | Multi-user |

Our dataset sits well below every threshold. Every PoC tool has a production equivalent with the same API surface — migration is infrastructure, not a code rewrite.

---

## Acknowledgements

Dataset: [H&M Personalized Fashion Recommendations — Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
