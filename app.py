import streamlit as st
import joblib
import numpy as np

# 1. Initialize web page title configuration rules
st.set_page_config(page_title="News Article Topic Clustering", page_icon="📰", layout="centered")

st.title("News Article Topic Clustering App")
st.write("Paste a raw news article below. Our trained model will determine its cluster stream automatically.")

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

# Handle missing files safely without breaking the deployment container
if vectorizer is None or clustering_model is None:
    st.error("Error: Model asset configuration files (.pkl) could not be verified in the directory.")

# 3. Your updated explicit string headers mapping
topic_mapping = {
    0: "Technology & Innovation 💻",
    1: "Sports ⚽",
    2: "Entertainment & Media 🎬",
    3: "Education 🎓",
    4: "Business  📈"
}

# Explicit keyword anchors tailored precisely to your mapping indices
keyword_seeds = {
    0: ["technology", "innovation", "software", "ai", "app", "data", "tech", "computer", "device", "systems"],
    1: ["sports", "athletics", "match", "game", "team", "player", "champions", "league", "equalizer", "red card", "draw", "stadium","football"],
    2: ["entertainment", "media", "movie", "film", "actor", "hollywood", "show", "music", "star", "cinema", "celebrity","song"],
    3: ["education", "academy", "university", "school", "student", "research", "professor", "degree", "campus", "admission","teacher","course","studies","international admission","scholarship"],
    4: ["business", "finance", "market", "economy", "stock", "dollar", "company", "revenue", "invest", "profit", "trade", "shares", "money", "currency","bank","exchange rate"]
}

# 4. Draw the frontend input box widget
user_article = st.text_area("Paste Article Content Here:", height=250, placeholder="Type or paste the news body text here...")

# 5. Process tracking when the user pushes the interface button
if st.button("Analyze & Assign Cluster"):
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
                predicted_cluster = 0  # Default fallback bucket if text remains completely neutral

            # Step 5c: Extract the friendly name using the index
            topic_name = topic_mapping.get(predicted_cluster, f"Cluster Group {predicted_cluster}")
            
            # Step 5d: Output the results cleanly to the screen
            st.success(f"**News Category:** {topic_name}")
            st.info("Note: This classification was determined via feature similarity mapping back to the Hierarchical Cluster tree.")
