import feedparser, xml.etree.ElementTree as ET, json, os, re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
OUTPUT_FILE = "temp.xml"
LAST_SEEN_FILE = "last_seen_temp.json"
THRESHOLD = 0.8
REPUTATION = [
    "The New York Times",
    "BBC",
    "Al Jazeera",
    "South China Morning Post",
    "The Hindu",
    "Asia Times",
    "Eurasia Review",
    "Middle East Eye",
    "Middle East Monitor",
    "The Moscow Times",
    "Financial Express",
    "United News Bangladesh",
    "The Business Standard",
    "The Daily Star",
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower().strip())

def get_source(entry):
    if "bbc" in entry.link: return "BBC"
    if "nytimes" in entry.link: return "The New York Times"
    if "aljazeera" in entry.link: return "Al Jazeera"
    if "scmp" in entry.link: return "South China Morning Post"
    if "hindu" in entry.link: return "The Hindu"
    if "asiatimes" in entry.link: return "Asia Times"
    if "eurasiareview" in entry.link: return "Eurasia Review"
    if "middleeasteye" in entry.link: return "Middle East Eye"
    if "middleeastmonitor" in entry.link: return "Middle East Monitor"
    if "moscowtimes" in entry.link: return "The Moscow Times"
    if "thefinancialexpress" in entry.link: return "Financial Express"
    if "tbsnews" in entry.link: return "The Business Standard"
    if "thedailystar" in entry.link: return "The Daily Star"
    if "unb" in entry.link: return "United News Bangladesh"
    return "Unknown"

# Load existing XML
if os.path.exists(OUTPUT_FILE):
    tree = ET.parse(OUTPUT_FILE)
    root = tree.getroot()
else:
    root = ET.Element("rss")
    channel = ET.SubElement(root, "channel")
    tree = ET.ElementTree(root)

# Load last seen
if os.path.exists(LAST_SEEN_FILE):
    last_seen = json.load(open(LAST_SEEN_FILE))
else:
    last_seen = {}

# Parse feeds
urls = [line.strip() for line in open(FEEDS_FILE) if line.strip() and not line.startswith("#")]
entries = []
for url in urls:
    feed = feedparser.parse(url)
    for e in feed.entries:
        title = e.get("title", "")
        link = e.get("link", "")
        if link in last_seen:  # skip old
            continue
        source = get_source(e)
        entries.append({"title": title, "link": link, "source": source})

# Cluster by title similarity
titles = [normalize(e["title"]) for e in entries]
if not titles:
    exit()

embeddings = model.encode(titles)
clusters = []
used = set()

for i, emb in enumerate(embeddings):
    if i in used: continue
    cluster = [i]
    for j in range(i+1, len(embeddings)):
        if cosine_similarity([emb], [embeddings[j]])[0][0] > THRESHOLD:
            cluster.append(j)
            used.add(j)
    clusters.append(cluster)

# Select best from cluster
def rank_source(source):
    return REPUTATION.index(source) if source in REPUTATION else len(REPUTATION)

selected = []
for cluster in clusters:
    cluster_entries = [entries[i] for i in cluster]
    best = sorted(cluster_entries, key=lambda x: rank_source(x["source"]))[0]
    selected.append(best)
    last_seen[best["link"]] = datetime.utcnow().isoformat()

# Append new to XML
channel = root.find("channel")
for item in selected:
    item_el = ET.SubElement(channel, "item")
    ET.SubElement(item_el, "title").text = item["title"]
    ET.SubElement(item_el, "link").text = item["link"]
    ET.SubElement(item_el, "source").text = item["source"]
    ET.SubElement(item_el, "pubDate").text = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
json.dump(last_seen, open(LAST_SEEN_FILE, "w"))
