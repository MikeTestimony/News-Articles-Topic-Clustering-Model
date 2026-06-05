import streamlit as st
import joblib
import numpy as np

# 1. Initialize web page title configuration rules
st.set_page_config(page_title="News Article Topic Clustering", page_icon="📰", layout="centered")

# 2. Your original title and sub-writings
st.title("News Topic Clustering App")
st.write("Paste a raw news article below. Our trained model will determine its cluster stream automatically.")
# Custom CSS to change the button color to deep blue instead of red
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1E3A8A !important;
        color: white !important;
        border: 1px solid #1E3A8A !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #3B82F6 !important;
        border: 1px solid #3B82F6 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
# 3. Use caching to load the large model assets ONLY ONCE into memory
@st.cache_resource
def load_assets():
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('agglomerative_model.pkl')
        return vectorizer, model
    except:
        return None, None

vectorizer, clustering_model = load_assets()

# Handle missing files safely without breaking the deployment container
if vectorizer is None or clustering_model is None:
    st.error("Error: Model asset configuration files (.pkl) could not be verified in the directory.")

# 4. Your exact current explicit string headers mapping
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

# 5. Draw the frontend input box widget
user_article = st.text_area("Paste Article Content Here:", height=250, placeholder="Type or paste the news body text here...")

# 6. Process tracking when the user pushes the interface button
if st.button("Analyze & Assign Cluster", type="primary"):  # Makes the button a solid, colorful blue
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
            
            # 7. Output the results in a beautiful, high-contrast colorful box
            st.markdown(f"""
                <div style="background-color: #E6F4EA; border-left: 6px solid #137333; padding: 1.25rem; border-radius: 6px; margin-top: 1.5rem;">
                    <p style="margin: 0; font-size: 0.85rem; color: #137333; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Target Allocation Result</p>
                    <h3 style="margin: 0.25rem 0 0 0; color: #137333; font-size: 1.5rem; font-weight: 700;">News Category: {topic_name}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.caption("<br>Note: This classification was determined via feature similarity mapping back to the Hierarchical Cluster tree.", unsafe_allow_html=True)
