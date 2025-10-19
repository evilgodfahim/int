import xml.etree.ElementTree as ET, json, os, re
from datetime import datetime
from collections import Counter

TEMP_FILE = "temp.xml"
OUTPUT_FILE = "final.xml"
LAST_SEEN_FILE = "last_seen_final.json"

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

def normalize(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower().strip())

def rank_source(source):
    return REPUTATION.index(source) if source in REPUTATION else len(REPUTATION)

if not os.path.exists(TEMP_FILE):
    exit()

tree = ET.parse(TEMP_FILE)
root = tree.getroot()
channel = root.find("channel")
items = channel.findall("item")

if os.path.exists(LAST_SEEN_FILE):
    last_seen = json.load(open(LAST_SEEN_FILE))
else:
    last_seen = {}

# Group by normalized title
groups = {}
for item in items:
    title = item.find("title").text or ""
    key = normalize(title)
    if key not in groups:
        groups[key] = []
    groups[key].append(item)

# Count how many feeds share similar title
counts = {k: len(v) for k, v in groups.items()}
sorted_titles = sorted(groups.keys(), key=lambda k: (-counts[k], rank_source(groups[k][0].find("source").text)))

# Keep top ones and prefer high repetition
selected = []
for key in sorted_titles:
    items_cluster = groups[key]
    best = sorted(items_cluster, key=lambda i: rank_source(i.find("source").text))[0]
    link = best.find("link").text
    if link not in last_seen:
        selected.append(best)
        last_seen[link] = datetime.utcnow().isoformat()

# Write final XML
rss = ET.Element("rss")
channel_out = ET.SubElement(rss, "channel")

for item in selected[:200]:  # keep top 200 max
    channel_out.append(item)

ET.ElementTree(rss).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
json.dump(last_seen, open(LAST_SEEN_FILE, "w"))
