# H&M Fashion Recommendation Dashboard

import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="H&M Recommendation System",
    layout="wide"
)

st.title("🛍️ H&M Personalized Fashion Recommendations")
st.write("Semantic similarity search displaying deduplicated products and all available colorways.")

# Path to your downloaded local images folder
LOCAL_IMAGE_DIR = "./images"

# A clean fallback placeholder image if an entry in the dataset is completely missing its image file
MISSING_IMAGE_PLACEHOLDER = "https://images.unsplash.com/photo-1540959733332-eab4deceeaf7?q=80&w=500&auto=format&fit=crop"

# ---------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)

@st.cache_data
def load_csv_data():
    try:
        df = pd.read_csv("articles.csv")
        df["article_id"] = df["article_id"].astype(int)
        return df.set_index("article_id")
    except Exception as e:
        st.error(f"Could not load articles.csv: {e}")
        return None

model = load_model()
client = get_qdrant_client()
articles_df = load_csv_data()

# ---------------------------------------------------
# Helper Function to Locate Local Images
# ---------------------------------------------------

def get_local_image_path(article_id):
    """
    Kaggle nests images in subfolders matching the first 3 characters of the ID.
    Example: article_id 108775015 -> path: ./images/010/0108775015.jpg
    """
    str_id = str(article_id).zfill(10)
    subfolder = str_id[:3]
    full_path = os.path.join(LOCAL_IMAGE_DIR, subfolder, f"{str_id}.jpg")
    
    if os.path.exists(full_path):
        return full_path
    return None

# ---------------------------------------------------
# User input
# ---------------------------------------------------

query_text = st.text_input(
    "Search fashion products",
    placeholder="Example: pants, black dress, hoodie"
)

# ---------------------------------------------------
# Similarity search & Presentation
# ---------------------------------------------------

if st.button("Find Similar Products", type="primary"):

    if query_text.strip() != "":
        with st.spinner("Searching database..."):
            query_vector = model.encode(query_text).tolist()

            results = client.query_points(
                collection_name="hm_articles",
                query=query_vector,
                limit=5,
                with_payload=True
            )

            st.subheader("🎯 Top Similar Products")

            if not results.points:
                st.info("No matching products found.")

            for index, hit in enumerate(results.points):
                payload = hit.payload if hit.payload else {}
                article_ids = payload.get("article_ids", [])

                if not article_ids:
                    continue

                # 1. Gather all unique available color labels for this product group from the CSV
                available_colors = []
                if articles_df is not None:
                    for a_id in article_ids:
                        a_id_int = int(a_id)
                        if a_id_int in articles_df.index:
                            row = articles_df.loc[a_id_int]
                            if isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                            
                            color = str(row.get("colour_group_name", "Unknown Color")).title()
                            pattern = str(row.get("graphical_appearance_name", "Solid")).title()
                            short_id = str(a_id)[-3:]
                            
                            # Construct a clean string for the list
                            label = f"**{color}** ({pattern}) — ID suffix: `#{short_id}`" if pattern != "Solid" else f"**{color}** — ID suffix: `#{short_id}`"
                            
                            if label not in available_colors:
                                available_colors.append(label)

                st.markdown("---")
                col1, col2 = st.columns([1.5, 3])

                # Use the primary (first) variant ID for main display and image rendering
                primary_id = int(article_ids[0])
                primary_row = articles_df.loc[primary_id] if (articles_df is not None and primary_id in articles_df.index) else {}
                if isinstance(primary_row, pd.DataFrame):
                    primary_row = primary_row.iloc[0]

                # 2. Rich Details Column
                with col2:
                    prod_name = str(primary_row.get("prod_name", "Fashion Item")).title()
                    product_type = str(primary_row.get("product_type_name", "Clothing Item")).title()
                    detail_desc = str(primary_row.get("detail_desc", "*No description available.*"))
                    
                    st.markdown(f"### {prod_name} <span style='font-size:14px; color:gray;'>({product_type})</span>", unsafe_allow_html=True)
                    st.write(detail_desc)
                    
                    # Layout Specifications for the main primary item
                    st.markdown("#### 🎨 Design & Demographics")
                    attr_col1, attr_col2 = st.columns(2)
                    
                    with attr_col1:
                        st.markdown(f"**Target Demographic:** `{primary_row.get('index_group_name', 'N/A')}`")
                        st.markdown(f"**Section:** {primary_row.get('section_name', 'N/A')}")
                        st.markdown(f"**Garment Type:** {primary_row.get('garment_group_name', 'N/A')}")
                        
                    with attr_col2:
                        st.markdown(f"**Primary Color Displayed:** **{primary_row.get('colour_group_name', 'N/A')}**")
                        st.markdown(f"**Primary Pattern:** {primary_row.get('graphical_appearance_name', 'N/A')}")
                        st.markdown(f"**Primary Article ID:** `{primary_id}`")

                    # 3. DISPLAY THE COLORS AS A LIST INSIDE THE PRODUCT CARD
                    st.markdown(f"#### 🏷️ Available Colors & Style Variations ({len(article_ids)} total items)")
                    if available_colors:
                        for color_item in available_colors:
                            st.markdown(f"* {color_item}")
                    else:
                        st.write("* No variant color metadata found in articles.csv.")

                # 4. Visual Sidebar Column
                with col1:
                    st.metric("Similarity Score", f"{hit.score * 100:.1f}%")
                    
                    # Look up primary image file locally
                    local_img_path = get_local_image_path(primary_id)
                    
                    if local_img_path:
                        st.image(local_img_path, use_container_width=True)
                    else:
                        st.image(MISSING_IMAGE_PLACEHOLDER, use_container_width=True)
                        st.caption("ℹ️ *Primary image missing in local subfolder.*")

    else:
        st.warning("Please enter a search query.")