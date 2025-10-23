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
WEIGHT_RECENCY = 2.0
KEYWORD_BOOST = 15.0

# Breaking news keywords
BREAKING_KEYWORDS = [
    "breaking", "urgent", "just in", "developing", "live",
    "war", "attack", "crisis", "death", "killed", "election",
    "breaking news", "update"
]

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

def has_breaking_keywords(title):
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in BREAKING_KEYWORDS)

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
    newest_date = max(a["pubDate"] for a in cluster)
    hours_old = (datetime.now(timezone.utc) - newest_date.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    recency_score = max(0, 24 - hours_old)
    breaking_boost = KEYWORD_BOOST if any(has_breaking_keywords(a["title"]) for a in cluster) else 0
    score = (
        unique_sources * WEIGHT_FEED_COUNT +
        avg_reputation * WEIGHT_REPUTATION +
        recency_score * WEIGHT_RECENCY +
        breaking_boost
    )
    return {
        "score": score,
        "feed_count": unique_sources,
        "avg_reputation": avg_reputation,
        "recency_score": recency_score,
        "has_breaking": breaking_boost > 0
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
    print(f"🔍 Filtering clusters (min {MIN_FEED_COUNT} feeds)...")

    important_clusters = []
    for cluster in clusters:
        if len(set(a["source"] for a in cluster)) >= MIN_FEED_COUNT:
            importance = calculate_importance(cluster)
            best_article = select_best_article(cluster)
            important_clusters.append({
                "article": best_article,
                "cluster_size": len(cluster),
                "importance": importance
            })

    print(f"✨ Found {len(important_clusters)} important stories")

    important_clusters.sort(key=lambda x: x["importance"]["score"], reverse=True)

    last_seen = load_last_seen()
    new_last_seen = dict(last_seen)
    final_articles = []

    for item in important_clusters:
        article = item["article"]
        if article["link"] not in last_seen:
            final_articles.append(item)
            new_last_seen[article["link"]] = datetime.now(timezone.utc).isoformat()
        if len(final_articles) >= MAX_FINAL_ARTICLES:
            break

    # Generate final.xml
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Fahim Final News Feed"
    ET.SubElement(channel, "link").text = "https://evilgodfahim.github.io/"
    ET.SubElement(channel, "description").text = "Curated important news from multiple sources"
    ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    for item in final_articles:
        article = item["article"]
        imp = item["importance"]
        xml_item = ET.SubElement(channel, "item")
        ET.SubElement(xml_item, "title").text = article["title"]
        ET.SubElement(xml_item, "link").text = article["link"]
        ET.SubElement(xml_item, "pubDate").text = article["pubDateStr"]
        source_text = f"{article['source']} (+{item['cluster_size']-1} other sources)" if item['cluster_size'] > 1 else article["source"]
        ET.SubElement(xml_item, "source").text = source_text
        desc = f"Importance: {imp['score']:.1f} | Covered by {imp['feed_count']} feeds | Reputation: {imp['avg_reputation']:.1f}"
        if imp['has_breaking']:
            desc += " | ⚡ Breaking"
        ET.SubElement(xml_item, "description").text = desc

    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ")
    tree.write(FINAL_XML_FILE, encoding="utf-8", xml_declaration=True)
    save_last_seen(new_last_seen)

    print(f"\n✅ Final feed generated: {FINAL_XML_FILE}")
    print(f"📝 Total stories: {len(final_articles)}")

if __name__ == "__main__":
    try:
        curate_final_feed()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)