import feedparser, xml.etree.ElementTree as ET, json, os, re
from datetime import datetime, timedelta, timezone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
OUTPUT_FILE = "temp.xml"
LAST_SEEN_FILE = "last_seen_temp.json"
THRESHOLD = 0.8

REPUTATION = [
    "The New York Times", "BBC", "Al Jazeera", "South China Morning Post",
    "The Hindu", "Asia Times", "Eurasia Review", "Middle East Eye",
    "Middle East Monitor", "The Moscow Times", "Financial Express",
    "United News Bangladesh", "The Business Standard", "The Daily Star"
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower().strip())

def get_source(entry):
    url = entry.get("link", "").lower()
    if "bbc" in url: return "BBC"
    if "nytimes" in url: return "The New York Times"
    if "aljazeera" in url: return "Al Jazeera"
    if "scmp" in url: return "South China Morning Post"
    if "hindu" in url: return "The Hindu"
    if "asiatimes" in url: return "Asia Times"
    if "eurasiareview" in url: return "Eurasia Review"
    if "middleeasteye" in url: return "Middle East Eye"
    if "middleeastmonitor" in url: return "Middle East Monitor"
    if "moscowtimes" in url: return "The Moscow Times"
    if "financialexpress" in url: return "Financial Express"
    if "tbsnews" in url: return "The Business Standard"
    if "thedailystar" in url: return "The Daily Star"
    if "unb" in url: return "United News Bangladesh"
    return "Unknown"

def parse_date(entry):
    if "published_parsed" in entry and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

# Load last seen
last_seen = json.load(open(LAST_SEEN_FILE)) if os.path.exists(LAST_SEEN_FILE) else {}

# Load existing XML
if os.path.exists(OUTPUT_FILE):
    tree = ET.parse(OUTPUT_FILE)
    root = tree.getroot()
else:
    root = ET.Element("rss")
    ET.SubElement(root, "channel")
    tree = ET.ElementTree(root)

urls = [l.strip() for l in open(FEEDS_FILE) if l.strip() and not l.startswith("#")]
entries = []
for url in urls:
    feed = feedparser.parse(url)
    for e in feed.entries:
        link = e.get("link", "")
        if not link or link in last_seen:
            continue
        entries.append({
            "title": e.get("title", ""),
            "link": link,
            "source": get_source(e),
            "date": parse_date(e)
        })

if not entries:
    exit()

# Cluster by title similarity
titles = [normalize(e["title"]) for e in entries]
embeddings = model.encode(titles)
used, clusters = set(), []

for i, emb in enumerate(embeddings):
    if i in used: continue
    cluster = [i]
    for j in range(i + 1, len(embeddings)):
        if cosine_similarity([emb], [embeddings[j]])[0][0] > THRESHOLD:
            cluster.append(j)
            used.add(j)
    clusters.append(cluster)

def rank_source(src): return REPUTATION.index(src) if src in REPUTATION else len(REPUTATION)

selected = []
for cluster in clusters:
    grouped = [entries[i] for i in cluster]
    best = sorted(grouped, key=lambda x: rank_source(x["source"]))[0]
    selected.append(best)
    last_seen[best["link"]] = datetime.utcnow().isoformat()

# Sort by date (newest first)
selected.sort(key=lambda x: x["date"], reverse=True)

channel = root.find("channel")
for e in selected:
    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = e["title"]
    ET.SubElement(item, "link").text = e["link"]
    ET.SubElement(item, "source").text = e["source"]
    ET.SubElement(item, "pubDate").text = e["date"].strftime("%a, %d %b %Y %H:%M:%S GMT")

tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
json.dump(last_seen, open(LAST_SEEN_FILE, "w")) 
