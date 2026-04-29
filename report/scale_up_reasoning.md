# Scale-Up Reasoning
## H&M In-Store Movement & Product Placement — Team GraphMinds

---

## 1. Tool Comparison Table

| Layer | Current Tool | Production-Scale Alternative | Why Switch | When to Switch |
|---|---|---|---|---|
| ETL | **DuckDB** | **Apache Spark** | DuckDB runs on one machine. At 500GB+ of transaction data across multiple years and regions, Spark distributes processing across a cluster — same SQL logic, 100x the throughput. The DuckDB-to-Spark migration is mostly a syntax change: `duckdb.execute(sql)` becomes `spark.sql(sql)`. | When data exceeds available RAM on a single machine, or when ETL must run on a schedule across multiple data sources simultaneously |
| Graph DB | **Neo4j Community** | **Neo4j Enterprise / AuraDB** | Community edition runs on one server with no horizontal scaling. Enterprise adds read replicas, sharding, and managed cloud hosting. At 1 billion+ nodes, a single Neo4j instance hits memory limits. | When graph exceeds ~500M nodes or requires 99.9% uptime SLA |
| ML | **scikit-learn** | **Spark MLlib / Ray** | scikit-learn trains on a single machine in memory. With 31M+ training rows and hundreds of features, training time becomes hours. Spark MLlib and Ray distribute training across nodes. | When training dataset exceeds available RAM or training time exceeds acceptable batch window |
| Vector Search | **Qdrant (local)** | **Qdrant Cloud / Pinecone** | Local Qdrant works for 105K article vectors. At 10M+ product descriptions across a global catalogue, a managed vector DB handles replication, index updates, and sub-10ms query SLA. | When vector count exceeds 1M or when sub-millisecond latency is required for live customer-facing search |
| Dashboard | **Streamlit (local)** | **Streamlit Cloud / Dash Enterprise** | Local Streamlit serves one user. A deployed version with caching, authentication, and load balancing serves hundreds of concurrent store managers across regions. | When more than 5 concurrent users need access, or when dashboard must be embedded in an existing retail management system |

---

## 2. The 5 Vs — H&M Dataset at Production Scale

| V | Current (This Project) | Production Scale | Implication |
|---|---|---|---|
| **Volume** | 28.7M in-store transactions, 105K articles, 1.37M customers — ~35 GB raw | A real H&M deployment would include all global markets: ~500M+ transactions per year, 500K+ articles across all regions, 50M+ customers | DuckDB replaced by Spark; Neo4j Community replaced by Enterprise with sharding |
| **Velocity** | Static batch — data downloaded once from Kaggle | Real-time POS data: every in-store purchase triggers an event within seconds; online purchases stream continuously | ETL pipeline becomes a streaming job (Kafka + Spark Streaming); graph updates become incremental MERGE operations rather than full reloads |
| **Variety** | Structured CSV only — transactions, articles, customers | Additional sources: RFID scan events, loyalty app GPS pings, social media sentiment, weather data, promotional calendar | Pipeline must handle JSON streams, image data (product photos for Qdrant), and semi-structured event logs alongside structured transaction CSVs |
| **Veracity** | Clean Kaggle dataset with known schema | Real POS data has duplicate scans, cancelled transactions, system outages, encoding errors, and schema changes mid-stream | DuckDB cleaning rules become more complex; COALESCE and CASE WHEN patterns scale directly but require continuous monitoring and alerting |
| **Value** | Academic proof of concept — demonstrates the method works | Commercial deployment: placement recommendations that increase revenue per sqm by 10-15% across 4,000+ H&M stores globally represents hundreds of millions in incremental annual revenue | The graph pipeline itself becomes a competitive asset; model retraining cadence shifts from once per project to weekly or daily |

---

## 3. Pipeline Limits

*To be completed in Session 7.*

---

## 4. Graph Scaling

*To be completed in Session 7.*

---

## 5. Batch vs Stream for Retail Domain

*To be completed in Session 7.*

---

## 6. Vector Search at Scale

*To be completed in Session 7.*

---

*Sections 3-6 will be finalised in Session 7 as per course requirements.*
