# Project Report — H&M Customer Intelligence Pipeline

**Team:** 5 Guys — Riya · Atharva · Farhan · Sacharith · Shubhendu  
**Dataset:** H&M Kaggle — 1.37M customers · 9.1M transactions · 105K products

---

## 1. Problem Statement

H&M needs to answer two operational questions from a single connected dataset: which customers are worth targeting with loyalty campaigns, and what products should be recommended to them. We built an end-to-end pipeline that answers both — a customer value classifier that identifies high-value customers (WHO to contact) and a semantic search engine that surfaces similar products by meaning (WHAT to recommend). Both are powered by the same knowledge graph.

The core hypothesis was that a customer's **structural position** in the purchase network carries predictive signal that flat tabular features cannot capture — and that adding graph-derived features would measurably improve a baseline classifier.

---

## 2. Methodology

### Data Pipeline

Raw CSV files were cleaned and deduplicated in **DuckDB**, then loaded into a **Neo4j** property graph with `Customer` and `Article` nodes connected by `PURCHASED` relationships. We added 109,684 `CO_PURCHASED` edges computed from transaction pairs. Uniqueness constraints on `customerId` and `articleId` prevented duplicate nodes during the `LOAD CSV` + `MERGE` process.

### Target Definition

The prediction target is binary: a customer who spent **above the median** in 2020 is `High-Value`, otherwise `Low-Value`. The dataset is heavily imbalanced — only 15.3% of customers are High-Value — which we handled with `class_weight='balanced'` in the classifier.

### Baseline Model

The baseline used six tabular features extracted via Cypher: all-time purchase count, 2020 purchase count, distinct sections visited, age, age band, and member status. These were fed into a scikit-learn `Pipeline` combining a `ColumnTransformer` (StandardScaler on numerical, OneHotEncoder on categorical) with a `RandomForestClassifier`. Crucially, the split happened **before** any preprocessing and the scaler lived **inside** the Pipeline — preventing data leakage by ensuring the scaler only ever saw training rows.

### Graph Enrichment

We ran two Neo4j GDS algorithms on a customer-article projection (PURCHASED edges, UNDIRECTED):

- **PageRank** (`dampingFactor=0.85`) — measures structural centrality. A customer connected to other high-purchasing customers scores higher than one with many low-activity connections.
- **Louvain community detection** (`maxLevels=10`) — groups customers with overlapping purchase histories into behavioural clusters.

Both columns were written back to Customer nodes, exported via Cypher, and merged onto the baseline feature matrix with a left join on `node_id` (= `customerId`). The same pipeline was then retrained with `pagerank` added as numerical and `community` as categorical.

### Semantic Search

In parallel, article `detail_desc` text was embedded with `all-MiniLM-L6-v2` (384-dimensional, cosine distance) and stored in a **Qdrant** vector collection. Descriptions were deduplicated before embedding so identical text produced a single vector, with all sharing `article_ids` retained in the payload.

---

## 3. Results

### Classifier Performance

| Metric | Baseline (degree only) | Enriched (+pagerank +community) | Delta |
|---|---|---|---|
| Accuracy | 81.02% | 86.65% | +5.62pp |
| F1 (macro) | 0.8049 | 0.8665 | +0.0616 |
| High-Value Precision | 0.73 | 0.87 | +14pp |
| High-Value Recall | 0.98 | 0.87 | −11pp |

The graph features delivered a meaningful **+5.62 percentage point accuracy gain** and a **+0.06 F1 lift**. The most business-relevant result is High-Value Precision rising from 0.73 to 0.87 — when the enriched model flags a customer as High-Value, it is correct 87% of the time.

The recall decrease from 0.98 to 0.87 is the expected precision-recall tradeoff: the enriched model became more selective, requiring both the right spending pattern and the right structural position before predicting High-Value. For a targeted campaign with a fixed budget, this precision gain is the correct trade — it eliminates far more false positives than the true positives it sacrifices.

### Graph Analytics Findings

**PageRank** revealed a 137× hub gap (max 25.34 vs median 0.185), confirming a small core of structurally central customers. Five of the top ten were in their 20s, identifying Young Fashion as the network core, with two unexpected 60+ hubs worth targeting.

**Louvain** found communities covering 43% of all customers in the top five clusters, with a modularity of 0.260 confirming real structure. The largest community (179,558 customers, 13.1%) was a Young Fashion core buyer segment, with 46% of its members in their 20s. Interpreting communities by their most frequent `storeSection` and `ageBand` turns the abstract integer labels into actionable personas — the basis for community-level CRM targeting rather than broad demographic blasts.

### Semantic Search

The search returned a top cosine similarity score of **0.89**, with short noun-phrase queries matching H&M's terse description style best. Deduplication was essential — without it, the top-5 results returned identical descriptions under different article IDs.

The dashboard implements the **Retrieve + Augment** halves of the RAG pattern: the user query is embedded and retrieved against Qdrant (Retrieve), and the dashboard presents the ranked results to the user who acts as the augmentation consumer (Augment). Generation (an LLM call) is deliberately scoped out for cost and reproducibility — the architecture would extend to full RAG by passing retrieved `raw_text` payloads into an LLM prompt.

---

## 4. Deliverable

A two-tab Streamlit dashboard ties everything together. Tab 1 provides natural-language product search returning ranked similar products with images and colour variants. Tab 2 offers customer-level value prediction with confidence scores, a top-10 PageRank leaderboard, and a community breakdown chart.

---

## 5. Conclusion

The project validated the central hypothesis: graph structure carries predictive signal beyond tabular features. Adding PageRank and community labels lifted classifier accuracy by 5.62 points and precision by 14 points with a clean 100% merge — proving the value of running graph analytics directly on Customer nodes rather than mismatched node types.

The key engineering lessons were the importance of preventing data leakage through the Pipeline abstraction, the necessity of deduplication before embedding, and the value of a clean `node_id` merge key carried end-to-end. Every tool in the stack (DuckDB, Neo4j, scikit-learn, Qdrant, Streamlit) sits well within its single-node scale ceiling, and each has a documented production equivalent with the same API surface — migration would be infrastructure, not a code rewrite.

It is worth stating that the measurement itself was the deliverable, not a guaranteed positive lift. Had the graph features produced no lift or a negative delta, that would have been an equally valid finding — it would have told us the prediction target was essentially local and that degree counts already captured the available signal. In our case the lift was both positive and stable, but the experimental discipline (a committed baseline, a single clean merge, a measured before/after) is what makes the result defensible.

The classifier tells you WHO to contact. The semantic search tells you WHAT to recommend. The graph makes both more accurate.
