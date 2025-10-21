import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ===== CONFIG =====
TEMP_XML_FILE = "temp.xml"
FINAL_XML_FILE = "final.xml"
LAST_SEEN_FILE = "last_seen.json"
FINAL_OCCURRENCE_THRESHOLD = 3   # must appear in >=3 feeds
MAX_ARTICLE_AGE_DAYS = 1         # skip articles older than 1 day
SIMILARITY_THRESHOLD = 0.50      # similarity for title clustering

# ===== MODEL =====
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== Load articles from temp.xml =====
def load_articles_from_temp():
    if not os.path.exists(TEMP_XML_FILE):
        return []
    tree = ET.parse(TEMP_XML_FILE)
    root = tree.getroot()
    articles = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source = item.findtext("source", "unknown").strip()
        if title and link:
            articles.append({
                "title": title,
                "link": link,
                "pubDate": pub_date,
                "source": source
            })
    return articles

# ===== Utility: Parse date =====
def parse_date(pub_date):
    try:
        return datetime.strptime(pub_date[:25], "%a, %d %b %Y %H:%M:%S")
    except Exception:
        return None

# ===== Utility: Check if article is recent =====
def is_recent(article):
    date = parse_date(article["pubDate"])
    if not date:
        return False
    return datetime.utcnow() - date < timedelta(days=MAX_ARTICLE_AGE_DAYS)

# ===== Load or create last_seen.json =====
def load_last_seen():
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, "r") as f:
            return json.load(f)
    return {}

def save_last_seen(data):
    with open(LAST_SEEN_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ===== Title clustering =====
def cluster_articles(articles):
    titles = [a["title"] for a in articles]
    embeddings = model.encode(titles)
    used = set()
    clusters = []

    for i, art in enumerate(articles):
        if i in used:
            continue
        group = [art]
        used.add(i)
        for j, other in enumerate(articles):
            if j in used:
                continue
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= SIMILARITY_THRESHOLD:
                group.append(other)
                used.add(j)
        clusters.append(group)
    return clusters

# ===== Reputation hierarchy =====
REPUTATION = [
    "nytimes", "bbc", "aljazeera", "thehindu", "scmp", "eurasia", "asiatimes",
    "moscowtimes", "middleeasteye", "middleeastmonitor", "daily star", "business standard",
    "bdnews24", "financial express", "united news bangladesh"
]

def get_reputation_score(source):
    s = source.lower()
    for i, name in enumerate(REPUTATION):
        if name in s:
            return len(REPUTATION) - i
    return 0

def select_representative_article(cluster):
    return sorted(cluster, key=lambda a: get_reputation_score(a["source"]), reverse=True)[0]

# ===== Generate final.xml =====
def generate_final():
    articles = load_articles_from_temp()
    if not articles:
        print("No articles found in temp.xml")
        return

    clusters = cluster_articles(articles)
    last_seen = load_last_seen()
    new_last_seen = dict(last_seen)

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Fahim Final Feed"
    ET.SubElement(channel, "link").text = "https://evilgodfahim.github.io/"
    ET.SubElement(channel, "description").text = "Most reported, reputable, recent stories."

    now = datetime.utcnow()
    ET.SubElement(channel, "pubDate").text = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

    count_included = 0
    for cluster in clusters:
        feeds_count = len({a["source"] for a in cluster})
        if feeds_count >= FINAL_OCCURRENCE_THRESHOLD:
            article = select_representative_article(cluster)
            if article["link"] in last_seen:
                continue
            if not is_recent(article):
                continue

            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = article["title"]
            ET.SubElement(item, "link").text = article["link"]
            ET.SubElement(item, "pubDate").text = article["pubDate"]
            ET.SubElement(item, "source").text = article["source"]
            new_last_seen[article["link"]] = article["pubDate"]
            count_included += 1

    tree = ET.ElementTree(rss)
    tree.write(FINAL_XML_FILE, encoding="utf-8", xml_declaration=True)
    save_last_seen(new_last_seen)
    print(f"✅ Final feed generated with {count_included} items.")

if __name__ == "__main__":
    generate_final()
