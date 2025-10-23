import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import json
import os
import re
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ===== CONFIG =====
TEMP_XML_FILE = "temp.xml"
FINAL_XML_FILE = "final.xml"
LAST_SEEN_FILE = "last_seen_final.json"

# Thresholds
MIN_FEED_COUNT = 3  # Story must appear in at least 3 feeds
SIMILARITY_THRESHOLD = 0.55  # Title clustering threshold
MAX_FINAL_ARTICLES = 500  # Maximum articles in final feed

# Importance scoring weights
WEIGHT_FEED_COUNT = 10.0
WEIGHT_REPUTATION = 0.5

# Source reputation hierarchy (higher = more reputable)
REPUTATION = {
    "The New York Times": 14,
    "BBC": 13,
    "Al Jazeera": 12,
    "The Hindu": 11,
    "South China Morning Post": 10,
    "Eurasia Review": 9,
    "Asia Times": 8,
    "The Moscow Times": 7,
    "Middle East Eye": 6,
    "Middle East Monitor": 5,
    "The Daily Star": 4,
    "The Business Standard": 3,
    "Financial Express": 2,
    "United News Bangladesh": 1,
}

# ===== MODEL =====
print("🔄 Loading embedding model...")
try:
    # Using MPNet for robust semantic equivalence
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("✅ Model loaded successfully (all-mpnet-base-v2)")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# ===== UTILITY FUNCTIONS =====
def normalize_title(title):
    """Normalize title for better clustering"""
    title = re.sub(r'\s+', ' ', title).strip()
    title = re.sub(r'[^\w\s\-\']', '', title)
    return title.lower()

def get_reputation_score(source):
    return REPUTATION.get(source, 0)

def parse_xml_date(date_str):
    try:
        return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
    except:
        try:
            return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S GMT")
        except:
            return datetime.now(timezone.utc)

# ===== ARTICLE LOADING =====
def load_articles_from_temp():
    if not os.path.exists(TEMP_XML_FILE):
        print(f"❌ {TEMP_XML_FILE} not found")
        return []

    tree = ET.parse(TEMP_XML_FILE)
    root = tree.getroot()
    articles = []

    for item in root.findall(".//item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date_str = item.findtext("pubDate", "").strip()
        source = item.findtext("source", "Unknown").strip()

        if not title or not link:
            continue

        pub_date = parse_xml_date(pub_date_str)

        articles.append({
            "title": title,
            "normalized_title": normalize_title(title),
            "link": link,
            "pubDate": pub_date,
            "pubDateStr": pub_date_str,
            "source": source
        })

    print(f"📥 Loaded {len(articles)} articles from temp.xml")
    return articles

# ===== CLUSTERING =====
def cluster_articles(articles):
    if not articles:
        return []

    print("🧠 Computing embeddings...")
    try:
        titles = [a["normalized_title"] for a in articles]
        embeddings = model.encode(titles, show_progress_bar=False)
        print(f"✅ Encoded {len(titles)} titles")
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
        return [[a] for a in articles]

    print("🔗 Clustering articles...")
    clusters = []
    used = set()

    for i, emb_i in enumerate(embeddings):
        if i in used:
            continue

        cluster = [articles[i]]
        used.add(i)

        for j in range(i + 1, len(embeddings)):
            if j in used:
                continue
            similarity = cosine_similarity([emb_i], [embeddings[j]])[0][0]
            if similarity >= SIMILARITY_THRESHOLD:
                cluster.append(articles[j])
                used.add(j)

        clusters.append(cluster)

    print(f"📊 Created {len(clusters)} clusters from {len(articles)} articles")
    return clusters

# ===== IMPORTANCE SCORING =====
def calculate_importance(cluster):
    unique_sources = len(set(a["source"] for a in cluster))
    reputations = [get_reputation_score(a["source"]) for a in cluster]
    avg_reputation = sum(reputations) / len(reputations) if reputations else 0
    
    score = (
        unique_sources * WEIGHT_FEED_COUNT +
        avg_reputation * WEIGHT_REPUTATION
    )
    
    return {
        "score": score,
        "feed_count": unique_sources,
        "avg_reputation": avg_reputation
    }

def select_best_article(cluster):
    sorted_cluster = sorted(
        cluster,
        key=lambda a: (get_reputation_score(a["source"]), a["pubDate"]),
        reverse=True
    )
    return sorted_cluster[0]

# ===== DEDUPLICATION =====
def load_last_seen():
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, "r") as f:
            data = json.load(f)
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            return {url: ts for url, ts in data.items() if datetime.fromisoformat(ts) > cutoff}
    return {}

def save_last_seen(data):
    with open(LAST_SEEN_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ===== MAIN CURATION =====
def curate_final_feed():
    articles = load_articles_from_temp()
    if not articles:
        print("⚠️  No articles to process")
        return

    clusters = cluster_articles(articles)
    print(f"🔍 Filtering clusters (min {MIN_FEED