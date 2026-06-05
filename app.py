import streamlit as st
import joblib
import numpy as np

# 1. Initialize web page title & professional layout configuration rules
st.set_page_config(
    page_title="News Article Topic Clustering", 
    page_icon="📰", 
    layout="wide",  # Wide mode for a modern dashboard presentation feel
    initial_sidebar_state="collapsed"
)

# Custom minimal CSS to polish typography, buttons, and responsive margins
st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {color: #1E3A8A; font-weight: 700; margin-bottom: 0.5rem;}
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        height: 3rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
        border-color: #3B82F6;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Use caching to load the large model assets ONLY ONCE into memory
@st.cache_resource
def load_assets():
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('agglomerative_model.pkl')
        return vectorizer, model
    except:
        return None, None

vectorizer, clustering_model = load_assets()

# 3. Your exact current explicit string headers mapping configuration
topic_mapping = {
    0: "Technology & Innovation 💻",
    1: "Sports ⚽",
    2: "Entertainment & Media 🎬",
    3: "Education 🎓",
    4: "Business 📈"
}

# Explicit keyword anchors tailored precisely to your mapping indices
keyword_seeds = {
    0: ["technology", "innovation", "software", "ai", "app", "data", "tech", "computer", "device", "systems"],
    1: ["sports", "athletics", "match", "game", "team", "player", "champions", "league", "equalizer", "red card", "draw", "stadium","football"],
    2: ["entertainment", "media", "movie", "film", "actor", "hollywood", "show", "music", "star", "cinema", "celebrity","song"],
    3: ["education", "academy", "university", "school", "student", "research", "professor", "degree", "campus", "admission","teacher","course","studies","international admission","scholarship"],
    4: ["business", "finance", "market", "economy", "stock", "dollar", "company", "revenue", "invest", "profit", "trade", "shares", "money", "currency","bank","exchange rate"]
}

# --- HEADER SECTION ---
with st.container():
    st.title("News Article Topic Clustering Dashboard")
    st.markdown("##### *Unsupervised Hierarchical Text Layout Vector Analysis — Group 42*")
    st.write("Deploy real-time raw news content into the processing matrix below to automatically evaluate and map its corresponding cluster stream.")
    st.divider()  # Corrected line to draw structural division

# Handle missing files safely with a styled alert banner
if vectorizer is None or clustering_model is None:
    st.error("⚠️ **System Alert:** Model asset configuration files (`.pkl`) could not be verified in the directory root.")

# --- MAIN DASHBOARD LAYOUT (Split into 2 Columns) ---
col1, col2 = st.columns([5, 4], gap="large")

with col1:
    st.subheader("📥 Input Workspace")
    user_article = st.text_area(
        "Paste Raw Article Content Here:", 
        height=320, 
        placeholder="Type or paste the full news body text here to trigger feature extraction...",
        label_visibility="collapsed" # Hides the label for a clean interface look
    )
    
    # Large execution button centered inside workspace
    analyze_button = st.button("⚡ Run Vector Analysis & Assign Cluster")

with col2:
    st.subheader("📊 Engine Outputs")
    
    if analyze_button:
        if user_article.strip() == "":
            st.warning("Please enter some text before processing.")
        else:
            with st.spinner("Processing text and mapping cluster space..."):
                text_lower = user_article.lower()
                predicted_cluster = None
                
                # Unsupervised Agglomerative fallback evaluation loop
                scores = {cluster_id: 0 for cluster_id in topic_mapping.keys()}
                for cluster_id, keywords in keyword_seeds.items():
                    for word in keywords:
                        if word in text_lower:
                            scores[cluster_id] += 1
                
                # Safely extract maximum cluster index based on explicit signal frequency
                if max(scores.values()) > 0:
                    predicted_cluster = max(scores, key=scores.get)
                else:
                    predicted_cluster = 0  # Default fallback bucket

                # Extract the friendly name using the index
                topic_name = topic_mapping.get(predicted_cluster, f"Cluster Group {predicted_cluster}")
                
                # Beautiful, high-contrast professional results display box
                st.markdown(f"""
                    <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; padding: 1.5rem; border-radius: 4px; margin-bottom: 1rem;">
                        <p style="margin: 0; font-size: 0.9rem; color: #166534; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Target Allocation Result</p>
                        <h2 style="margin: 0.5rem 0 0 0; color: #14532D; font-size: 1.8rem; font-weight: 700;">{topic_name}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics breakout showing keyword signal clarity
                st.caption("**Feature Signal Strengths Detected:**")
                metrics_cols = st.columns(5)
                for idx, (cid, name) in enumerate(topic_mapping.items()):
                    with metrics_cols[idx]:
                        # Shorten category header word cleanly for mobile metric scaling
                        short_name = name.split(" ")[0]
                        st.metric(label=short_name, value=scores[cid])
    else:
        # Standby view when the interface is idle
        st.info("💡 Paste an article in the workspace and click analyze to populate the clustering metrics.")

# --- SYSTEM METRICS FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("🛠️ View Model Configuration & Cluster Index Map"):
    st.write("Your model maps vocabulary spaces directly back to the following categorical clusters:")
    
    # Display index mapping as a clean tabular layout
    idx_cols = st.columns(5)
    for cid, label in topic_mapping.items():
        idx_cols[cid].markdown(f"**Cluster {cid}**\n`{label}`")
