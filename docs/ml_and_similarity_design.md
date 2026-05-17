# ML and Similarity Design Document
**Team:** GraphMinds — Riya · Atharva · Farhan · Sacharith · Shubhendu
**Course:** Big Data and Business Intelligence Capstone
**Last updated:** Session 6

---

## Section 1: Prediction Task

**Target:** `High-Value` vs `Low-Value` customer classification (binary)

**Rule:** A customer who spent **above the median** in 2020 = `High-Value`. At or below median, or with no 2020 purchases = `Low-Value`.

**Consumer:** A Store Manager at H&M uses the predicted customer value tier to decide which customers to target with promotions, loyalty rewards, and personalised outreach.

**One-sentence framing:**
> A Store Manager uses the predicted customer value tier to decide which customers to prioritise for H&M loyalty campaigns.

---

## Section 2: Text Field for Embedding

**Field chosen:** `detail_desc` on the Article node — the product description text.

**Why this field:**
- Richest free-text field in the dataset — contains semantic product information (fabric, style, fit, occasion)
- `productType` and `storeSection` are structured categorical fields — no benefit from embedding
- `name` is too short (2–4 words) for meaningful sentence-level embeddings
- `detail_desc` averages 15–30 words — fits whole-field embedding with no chunking needed

**Chunking decision:** Whole-field embedding. Each article description fits within `all-MiniLM-L6-v2`'s 256-token context window.

**Embedding model:** `all-MiniLM-L6-v2` — 384-dimensional output, cosine distance, CPU-friendly.

---

## Section 3: Streamlit Similarity Widget — User Journey

The store manager opens the Streamlit dashboard and navigates to the **Product Search** tab. They type a free-text description of a product — for example, *"slim fit dark wash denim jeans for women"*. The dashboard embeds the query using the same `all-MiniLM-L6-v2` model used at index time, queries the Qdrant `hm_articles` collection for the top-N nearest neighbours by cosine similarity (N selectable: 5, 10, or 15), and renders a results card for each match showing the product name, type, colour variants, garment group, section, and similarity score as a percentage. Each card also displays a local product image where available. Results are ranked by cosine similarity score descending. The manager uses the results to identify similar products already in the catalog and make informed restocking or co-placement decisions.

---

## Section 4: Graph-Analytics Columns S6 Will Add

**Merge key:** `node_id` = `customerId` on Customer nodes.
**Node type:** Customer — same as the S5 classifier unit. No node-type mismatch.

### PageRank on Customer Nodes
- **Column name:** `pagerank`
- **Graph projected:** Customer + Article nodes, PURCHASED edges, UNDIRECTED
- **Written by:** `gds.pageRank.write('customer_graph', {writeProperty: 'pagerank', maxIterations: 20, dampingFactor: 0.85})`
- **Interpretation:** Measures how structurally central each customer is in the purchase network. A high-PageRank customer bought the same products as other high-PageRank customers — they are embedded in the core of the purchase network.
- **Predicted lift direction:** Positive. Central customers are likely High-Value — they purchase popular hub products that connect many customer segments. This adds global structural signal that local degree counts cannot capture.

### Louvain Community on Customer Nodes
- **Column name:** `community`
- **Written by:** `gds.louvain.write('customer_graph', {writeProperty: 'community', maxLevels: 10, tolerance: 0.0001})`
- **Interpretation:** Groups customers with overlapping purchase histories into communities. Each community may represent a purchasing persona — e.g. sportswear buyers, Young Fashion buyers, Womenswear basics buyers.
- **Predicted lift direction:** Positive. High-Value customers likely cluster into communities with other High-Value customers (they buy similar premium or high-frequency products). Community membership adds a persona-level signal beyond individual degree counts.

### Before / After Metric (S6 result)

| Metric | S5 Baseline (degree only) | S6 Enriched (+pagerank +community) | Delta |
|---|---|---|---|
| Accuracy | *(fill from Cell 12 output)* | *(fill from Cell 12 output)* | *(fill from Cell 12 output)* |
| F1 (macro) | *(fill from Cell 12 output)* | *(fill from Cell 12 output)* | *(fill from Cell 12 output)* |

---

## Section 5: Knowledge Graph Schema (Final)

### Node Labels and Key Properties

| Label | Key Property | Other Properties |
|---|---|---|
| `Customer` | `customerId` | `age`, `ageBand`, `memberStatus`, `pagerank`*, `community`* |
| `Article` | `articleId` | `name`, `productType`, `garmentGroup`, `storeSection`, `detail_desc` *(embedded into Qdrant `hm_articles` collection)* |
| `StoreSection` | `storeSectionId` | `name` |
| `Department` | `departmentId` | `name`, `indexGroup` |
| `ProductGroup` | `productGroupId` | `name` |

*Derived properties written by Neo4j GDS in Session 6 onto **Customer** nodes.

### Relationship Types

| Type | From → To | Key Properties |
|---|---|---|
| `PURCHASED` | Customer → Article | `txDate`, `price`, `yearMonth` |
| `CO_PURCHASED` | Article → Article | `timesBoughtTogether`, `supportScore` |
| `IN_SECTION` | Article → StoreSection | — |
| `BELONGS_TO_DEPT` | Article → Department | — |
| `IN_GROUP` | Article → ProductGroup | — |

### GDS Projection Used in S6
```
Nodes:         Customer, Article
Relationships: PURCHASED (UNDIRECTED)
Name:          customer_graph
Purpose:       Connect customers via shared article purchases
               Customer → Article ← Customer = implicit customer-customer link
```

### Constraints
```cypher
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer)     REQUIRE c.customerId  IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article)      REQUIRE a.articleId   IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:StoreSection) REQUIRE s.storeSectionId IS UNIQUE;
```

---

## Section 6: Streamlit Data Flow

### Load-time queries (run once when dashboard starts)
1. Pull top-10 customers by `pagerank` → populate **Top Customers** panel (Tab 2)
2. Load `customer_value_classifier_s6.pkl` into memory → ready for Customer Lookup prediction
3. Load `all-MiniLM-L6-v2` model into memory → ready for similarity search
4. Load `articles.csv` into memory → ready for product metadata display

### User-input queries (run on each user interaction)
5. **Product Search (Tab 1):** User types product description → embed with `all-MiniLM-L6-v2` → `client.query_points(collection='hm_articles', query=vec, limit=N)` → render top-N similar articles with scores, colour variants, and local images
6. **Customer Lookup (Tab 2):** User inputs `customerId` → Cypher query pulls degree features + `pagerank` + `community` → model predicts value tier → show High-Value / Low-Value with confidence scores and full customer profile
7. **Top Customers (Tab 2):** Button click → Cypher `ORDER BY pagerank DESC LIMIT 10` → render table
8. **Community Breakdown (Tab 2):** Button click → Cypher `GROUP BY community ORDER BY size DESC LIMIT 5` → render table + bar chart

### Background (static, no refresh needed)
9. Qdrant `hm_articles` collection — pre-indexed at Session 5, read-only at runtime
10. Neo4j graph — read-only at runtime (GDS writes done in Session 6, not re-run in dashboard)

---

## Scale-Up Reasoning

| Tool | PoC (Capstone) | Production Alternative | Threshold |
|---|---|---|---|
| ETL | DuckDB | Apache Spark | ~100 GB |
| Graph DB | Neo4j Community | Neo4j Enterprise | >10B relationships |
| ML | scikit-learn | Spark MLlib | ~100 GB |
| Vectors | Qdrant local Docker | Qdrant Cloud | ~few million vectors |
| Dashboard | Streamlit local | Streamlit Cloud | Multi-user |

**ML scale-up sentence:**
Our working set is well under 1 GB — two orders of magnitude below the ~100 GB scikit-learn ceiling — so scikit-learn dominates on every dimension (microsecond predictions, no cluster ops, full algorithm breadth). We would migrate to Spark MLlib once the working set crosses ~50 GB, which is far above any plausible growth path for this dataset.

---

## Key Design Decision: Customer-Level PageRank

Previous attempt ran PageRank on Article nodes and merged onto Customer rows — a node-type mismatch that caused most merge rows to be NaN and produced a negative delta.

This version runs PageRank and Louvain directly on **Customer nodes** via **PURCHASED relationships**. The `node_id` = `customerId` on both the S5 feature matrix and the GDS export, guaranteeing a clean one-to-one merge. The business logic is also stronger: a customer's structural centrality in the purchase network (PageRank) and their purchasing persona cluster (Louvain) are direct signals for customer value prediction.
