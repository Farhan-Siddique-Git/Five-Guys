# H&M Fashion Recommendation Dashboard — Two-Tab Version
# Tab 1: Semantic Product Search
# Tab 2: Customer Value Lookup (S6 enriched model)

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="H&M Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR      = Path(r"C:/Jupyter/abc/models")
LOCAL_IMAGE_DIR = r"C:/Jupyter/abc/session 7/images"
MISSING_IMAGE   = "https://images.unsplash.com/photo-1540959733332-eab4deceeaf7?q=80&w=500&auto=format&fit=crop"

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "instance1")

# ── ALL function definitions FIRST ────────────────────────────────────────────

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant():
    return QdrantClient(host="localhost", port=6333)

@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@st.cache_resource
def load_classifier():
    try:
        path = MODELS_DIR / "customer_value_classifier_s6.pkl"
        if path.exists():
            return joblib.load(path)
        path_s5 = MODELS_DIR / "customer_value_classifier.pkl"
        if path_s5.exists():
            return joblib.load(path_s5)
        return None
    except Exception as e:
        st.error(f"Classifier load error: {e}")
        return None

@st.cache_data
def load_articles_csv():
    try:
        df = pd.read_csv("articles.csv")
        df["article_id"] = df["article_id"].astype(int)
        return df.set_index("article_id")
    except Exception as e:
        st.error(f"Could not load articles.csv: {e}")
        return None

def get_local_image_path(article_id):
    str_id    = str(article_id).zfill(10)
    subfolder = str_id[:3]
    full_path = os.path.join(LOCAL_IMAGE_DIR, subfolder, f"{str_id}.jpg")
    return full_path if os.path.exists(full_path) else None

def run_cypher(query, params={}):
    with neo4j_driver.session() as session:
        return session.run(query, params).data()

def fetch_customer_features(customer_id: str):
    query = """
    MATCH (c:Customer {customerId: $cid})

    OPTIONAL MATCH (c)-[p:PURCHASED]->()
    WITH c, COUNT(p) AS purchase_degree

    OPTIONAL MATCH (c)-[p2020:PURCHASED]->()
    WHERE p2020.yearMonth STARTS WITH '2020'
    WITH c, purchase_degree, COUNT(p2020) AS purchase_degree_2020

    OPTIONAL MATCH (c)-[ps:PURCHASED]->(a:Article)
    WHERE ps.yearMonth STARTS WITH '2020'
    WITH c, purchase_degree, purchase_degree_2020,
         COUNT(DISTINCT a.storeSection) AS sections_visited

    RETURN
        c.customerId   AS node_id,
        c.age          AS age,
        c.ageBand      AS ageBand,
        c.memberStatus AS memberStatus,
        c.pagerank     AS pagerank,
        c.community    AS community,
        purchase_degree,
        purchase_degree_2020,
        sections_visited
    """
    result = run_cypher(query, {"cid": customer_id})
    return result[0] if result else None

def fetch_top_pagerank(limit=10):
    query = """
    MATCH (c:Customer)
    WHERE c.pagerank IS NOT NULL
    RETURN c.customerId AS customer_id,
           c.pagerank   AS pagerank,
           c.community  AS community,
           c.ageBand    AS age_band
    ORDER BY pagerank DESC
    LIMIT $limit
    """
    return run_cypher(query, {"limit": limit})

def fetch_community_summary():
    query = """
    MATCH (c:Customer)
    WHERE c.community IS NOT NULL
    RETURN c.community AS community,
           COUNT(c)    AS size
    ORDER BY size DESC
    LIMIT 5
    """
    return run_cypher(query)

# ── THEN call them ─────────────────────────────────────────────────────────────
embedding_model = load_embedding_model()
qdrant_client   = get_qdrant()
neo4j_driver    = get_neo4j_driver()
classifier      = load_classifier()
articles_df     = load_articles_csv()

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .high-value {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 4px solid #28a745;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #155724;
    }
    .low-value {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border-left: 4px solid #ffc107;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛍️ H&M Intelligence Dashboard")
st.caption("Semantic product search · Customer value prediction · Graph analytics")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Product Search", "👤  Customer Loyalty"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRODUCT SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Semantic Similarity Search")
    st.write("Find products by describing what you're looking for in natural language.")

    query_text = st.text_input(
        "Search fashion products",
        placeholder="e.g. black floral summer dress, oversized hoodie, slim fit chinos",
        key="search_input"
    )

    col_search, col_limit = st.columns([3, 1])
    with col_limit:
        result_limit = st.selectbox("Results", [5, 10, 15], index=0)

    if st.button("Find Similar Products", type="primary", key="search_btn"):
        if query_text.strip():
            with st.spinner("Searching..."):
                query_vector = embedding_model.encode(query_text).tolist()
                results = qdrant_client.query_points(
                    collection_name="hm_articles",
                    query=query_vector,
                    limit=result_limit,
                    with_payload=True
                )

            st.subheader(f"🎯 Top {result_limit} results for: *{query_text}*")

            if not results.points:
                st.info("No matching products found.")

            for hit in results.points:
                payload     = hit.payload or {}
                article_ids = payload.get("article_ids", [])
                if not article_ids:
                    continue

                available_colors = []
                if articles_df is not None:
                    for a_id in article_ids:
                        a_id_int = int(a_id)
                        if a_id_int in articles_df.index:
                            row = articles_df.loc[a_id_int]
                            if isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                            color    = str(row.get("colour_group_name", "Unknown")).title()
                            pattern  = str(row.get("graphical_appearance_name", "Solid")).title()
                            short_id = str(a_id)[-3:]
                            label = (
                                f"**{color}** ({pattern}) — `#{short_id}`"
                                if pattern != "Solid"
                                else f"**{color}** — `#{short_id}`"
                            )
                            if label not in available_colors:
                                available_colors.append(label)

                st.markdown("---")
                col1, col2 = st.columns([1.5, 3])

                primary_id  = int(article_ids[0])
                primary_row = {}
                if articles_df is not None and primary_id in articles_df.index:
                    primary_row = articles_df.loc[primary_id]
                    if isinstance(primary_row, pd.DataFrame):
                        primary_row = primary_row.iloc[0]

                with col2:
                    prod_name   = str(primary_row.get("prod_name", "Fashion Item")).title()
                    prod_type   = str(primary_row.get("product_type_name", "Clothing")).title()
                    detail_desc = str(primary_row.get("detail_desc", "*No description available.*"))

                    st.markdown(
                        f"### {prod_name} <span style='font-size:14px;color:gray;'>({prod_type})</span>",
                        unsafe_allow_html=True
                    )
                    st.write(detail_desc)

                    st.markdown("#### 🎨 Design & Demographics")
                    a1, a2 = st.columns(2)
                    with a1:
                        st.markdown(f"**Target:** `{primary_row.get('index_group_name', 'N/A')}`")
                        st.markdown(f"**Section:** {primary_row.get('section_name', 'N/A')}")
                        st.markdown(f"**Garment:** {primary_row.get('garment_group_name', 'N/A')}")
                    with a2:
                        st.markdown(f"**Colour:** **{primary_row.get('colour_group_name', 'N/A')}**")
                        st.markdown(f"**Pattern:** {primary_row.get('graphical_appearance_name', 'N/A')}")
                        st.markdown(f"**Article ID:** `{primary_id}`")

                    st.markdown(f"#### 🏷️ Available Variants ({len(article_ids)} total)")
                    for c in available_colors:
                        st.markdown(f"* {c}")

                with col1:
                    st.metric("Similarity", f"{hit.score * 100:.1f}%")
                    img_path = get_local_image_path(primary_id)
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    else:
                        st.image(MISSING_IMAGE, use_container_width=True)
                        st.caption("ℹ️ Image not in local folder.")
        else:
            st.warning("Please enter a search query.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER LOYALTY
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Customer Value Prediction")
    st.write(
        "Look up any customer by ID to see their predicted value tier, "
        "PageRank centrality, and community segment."
    )

    if classifier is None:
        st.error(
            "No classifier found. Make sure `customer_value_classifier_s6.pkl` "
            "exists in the models directory."
        )
    else:
        st.markdown("#### 🔎 Individual Customer Lookup")
        customer_id_input = st.text_input(
            "Enter Customer ID",
            placeholder="Paste a customerId hash here...",
            key="customer_input"
        )

        if st.button("Predict Customer Value", type="primary", key="predict_btn"):
            if customer_id_input.strip():
                with st.spinner("Fetching customer data from Neo4j..."):
                    features = fetch_customer_features(customer_id_input.strip())

                if features is None:
                    st.error("Customer not found. Check the ID and try again.")
                else:
                    row = pd.DataFrame([{
                        "purchase_degree":      features.get("purchase_degree", 0) or 0,
                        "purchase_degree_2020": features.get("purchase_degree_2020", 0) or 0,
                        "sections_visited":     features.get("sections_visited", 0) or 0,
                        "age":                  features.get("age", 0) or 0,
                        "pagerank":             features.get("pagerank", 0.0) or 0.0,
                        "ageBand":              features.get("ageBand", "Unknown") or "Unknown",
                        "memberStatus":         features.get("memberStatus", "Unknown") or "Unknown",
                        "community":            str(features.get("community", "-1") or -1),
                    }])

                    prediction    = classifier.predict(row)[0]
                    proba         = classifier.predict_proba(row)[0]
                    proba_classes = classifier.classes_
                    confidence    = dict(zip(proba_classes, proba))

                    st.markdown("---")
                    pred_col, stats_col = st.columns([1, 2])

                    with pred_col:
                        if prediction == "High-Value":
                            st.markdown(
                                '<div class="high-value">⭐ High-Value Customer</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="low-value">🔵 Low-Value Customer</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.metric("High-Value confidence", f"{confidence.get('High-Value', 0) * 100:.1f}%")
                        st.metric("Low-Value confidence",  f"{confidence.get('Low-Value', 0) * 100:.1f}%")

                    with stats_col:
                        st.markdown("#### 📊 Customer Profile")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("All-time purchases", f"{features.get('purchase_degree', 0):,}")
                        m2.metric("2020 purchases",     f"{features.get('purchase_degree_2020', 0):,}")
                        m3.metric("Sections visited",   f"{features.get('sections_visited', 0):,}")

                        m4, m5, m6 = st.columns(3)
                        m4.metric("Age band",       str(features.get("ageBand", "N/A")))
                        m5.metric("Member status",  str(features.get("memberStatus", "N/A")))
                        m6.metric("PageRank score", f"{features.get('pagerank', 0.0):.4f}")

                        st.markdown(
                            f"**Community segment:** `{features.get('community', 'N/A')}`  \n"
                            f"**Customer ID:** `{features.get('node_id', customer_id_input)}`"
                        )
            else:
                st.warning("Please enter a customer ID.")

        st.divider()

        # ── Top 10 by PageRank ────────────────────────────────────────────────
        st.markdown("#### 🏆 Top 10 Customers by PageRank")
        st.caption("Most structurally central buyers in the H&M purchase network.")

        if st.button("Load Top Customers", key="top_pr_btn"):
            with st.spinner("Querying Neo4j..."):
                top_customers = fetch_top_pagerank(limit=10)

            if top_customers:
                top_df = pd.DataFrame(top_customers)
                top_df.index = range(1, len(top_df) + 1)
                top_df.columns = ["Customer ID", "PageRank", "Community", "Age Band"]
                top_df["PageRank"] = top_df["PageRank"].round(4)
                top_df["Customer ID"] = top_df["Customer ID"].astype(str).str[:20] + "..."
                st.dataframe(top_df, use_container_width=True)
            else:
                st.info("No PageRank data found. Run gds.pageRank.write in 03_graph_analytics.ipynb first.")

        st.divider()

        # ── Community breakdown ───────────────────────────────────────────────
        st.markdown("#### 🗂️ Top 5 Customer Communities")
        st.caption("Clusters of customers with overlapping purchase histories.")

        if st.button("Load Community Breakdown", key="community_btn"):
            with st.spinner("Querying Neo4j..."):
                communities = fetch_community_summary()

            if communities:
                comm_df = pd.DataFrame(communities)
                comm_df.columns = ["Community ID", "Customer Count"]
                comm_df["% of shown"] = (
                    comm_df["Customer Count"] / comm_df["Customer Count"].sum() * 100
                ).round(1).astype(str) + "%"
                st.dataframe(comm_df, use_container_width=True)
                st.bar_chart(comm_df.set_index("Community ID")["Customer Count"])
            else:
                st.info("No community data found. Run gds.louvain.write in 03_graph_analytics.ipynb first.")
