# Scale-Up Reasoning
**Team:** GraphMinds — Riya · Atharva · Farhan · Sacharith · Shubhendu
**Course:** Big Data and Business Intelligence Capstone
**Last updated:** Session 5

---

## Tool Selection Map

| Tool | PoC (Capstone) | Production Alternative | Threshold | Trade-off |
|---|---|---|---|---|
| ETL | DuckDB | Apache Spark | ~100 GB | Spark adds cluster ops + job overhead; DuckDB wins at single-node scale |
| Graph DB | Neo4j Community | Neo4j Enterprise / AuraDB | >10B relationships | Enterprise adds clustering, multi-tenancy, HA; overkill at capstone scale |
| ML | scikit-learn | Spark MLlib | ~100 GB | In-process microsecond predicts vs distributed scale; cluster ops cost; algorithm breadth narrows |
| Vectors | Qdrant local Docker | Qdrant Cloud | ~few million vectors | Managed sharding + replication; subscription cost; zero-ops API surface |
| Dashboard | Streamlit local | Streamlit Cloud | Multi-user / production | Cloud handles auth, scaling, secrets; local is sufficient for PoC demo |

---

## ETL Row — DuckDB vs Apache Spark

**PoC choice:** DuckDB
**Production alternative:** Apache Spark
**Threshold:** ~100 GB working set

Our raw CSV files total under 4 GB — two orders of magnitude below the ~100 GB DuckDB single-node ceiling. DuckDB processes the full dataset in seconds with zero infrastructure overhead. We would migrate the ETL pipeline to Apache Spark once the working set crosses ~50 GB, which is well above any plausible growth path for the H&M transaction dataset in this course context.

---

## ML Row — scikit-learn vs Spark MLlib

**PoC choice:** scikit-learn
**Production alternative:** Spark MLlib
**Threshold:** ~100 GB working set

| Dimension | scikit-learn | Spark MLlib |
|---|---|---|
| Data volume | < 100 GB single-node, fits in memory | 100 GB – multi-TB distributed |
| API breadth | Wide (classifiers, regressors, transformers, pipelines, model selection) | Narrower (focused on distributed-friendly algorithms; no SVM beyond linear, no t-SNE) |
| Algorithm depth | Mature, dozens of variants per family | Adequate for production-shaped problems |
| Deployment cost | `pip install`, runs on a laptop | Cluster (YARN / K8s / Standalone), real ops |
| Latency at predict time | Microseconds (in-process) | Seconds-to-minutes per Spark job |
| Learning curve | Low | Moderate (Spark fundamentals first) |

**Defend sentence:**
Our working set is well under 1 GB — two orders of magnitude below the ~100 GB scikit-learn ceiling — so scikit-learn dominates on every dimension (microsecond predictions, no cluster ops, full algorithm breadth). We would migrate to Spark MLlib once the working set crosses ~50 GB, which is far above any plausible growth path for this dataset.

---

## Vector Search Row — Qdrant Local vs Qdrant Cloud

**PoC choice:** Qdrant local Docker
**Production alternative:** Qdrant Cloud
**Threshold:** ~few million vectors or multi-tenant deployment

The H&M articles collection contains ~43,000 unique description vectors at 384 dimensions — well within the single-container Qdrant ceiling. Sub-millisecond HNSW queries are achievable locally with no operational overhead. We would migrate to Qdrant Cloud once the vector count exceeds a few million, or when the dashboard moves to multi-user production deployment requiring managed sharding, replication, and zero-ops scaling.

**Trade-off:** Qdrant Cloud adds managed cluster ops, horizontal sharding, and replication at a subscription cost. The local Docker deployment is sufficient for the PoC and reproducible from a single `docker run` command.

---

## Why Vector Search Adds Value Here

The H&M `detail_desc` field contains thousands of free-text product descriptions written in varied language — different authors describing similar products using different vocabulary ("slim-fit trousers" vs "tapered leg chinos"). Semantic similarity search clusters these where keyword search would fragment them. This is the canonical good fit: large catalog, varied authorship, cross-vocabulary user queries.

**Chunking decision:** Whole-field embedding. Each `detail_desc` averages 15–30 words, well within `all-MiniLM-L6-v2`'s 256-token context window. No chunking required.

---

## RAG Scope

The capstone implements **R + A** (Retrieve + Augment) only. Generation (G) is conceptual:

- **Retrieve:** User query → embed with `all-MiniLM-L6-v2` → `client.query_points()` → top-5 similar articles
- **Augment:** Streamlit renders the retrieved articles alongside their similarity scores, section, and colour variants — the user is the augmentation consumer
- **Generate (conceptual only):** The architecture would extend to full RAG by passing retrieved chunks as context to an LLM; scoped out for cost and reproducibility reasons in this PoC

The G step could be added in approximately two hours of additional work using the Anthropic or OpenAI API, with the retrieved `raw_text` payloads injected into the prompt template.
