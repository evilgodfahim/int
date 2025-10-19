import xml.etree.ElementTree as ET, json, os, re
from datetime import datetime, timedelta, timezone
from collections import defaultdict

TEMP_FILE = "temp.xml"
OUTPUT_FILE = "final.xml"
LAST_SEEN_FILE = "last_seen_final.json"

REPUTATION = [
    "The New York Times", "BBC", "Al Jazeera", "South China Morning Post",
    "The Hindu", "Asia Times", "Eurasia Review", "Middle East Eye",
    "Middle East Monitor", "The Moscow Times", "Financial Express",
    "United News Bangladesh", "The Business Standard", "The Daily Star"
]

def normalize(t): return re.sub(r"[^a-zA-Z0-9 ]", "", t.lower().strip())
def rank_source(s): return REPUTATION.index(s) if s in REPUTATION else len(REPUTATION)

if not os.path.exists(TEMP_FILE):
    exit()

tree = ET.parse(TEMP_FILE)
root = tree.getroot()
channel = root.find("channel")
items = channel.findall("item")

last_seen = json.load(open(LAST_SEEN_FILE)) if os.path.exists(LAST_SEEN_FILE) else {}
now = datetime.now(timezone.utc)

# Group similar titles
groups = defaultdict(list)
for it in items:
    title = it.findtext("title", "")
    key = normalize(title)
    groups[key].append(it)

# Filter clusters by repetition + recency
selected = []
for key, cluster in groups.items():
    # Parse pubDate
    best = sorted(cluster, key=lambda x: rank_source(x.findtext("source", "")))[0]
    pub_str = best.findtext("pubDate")
    try:
        pub_date = datetime.strptime(pub_str, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
    except:
        pub_date = now

    if (now - pub_date) > timedelta(days=1):  # older than 1 day → skip
        continue

    link = best.findtext("link", "")
    if link not in last_seen:
        selected.append(best)
        last_seen[link] = now.isoformat()

# Sort by pubDate (latest first)
selected.sort(key=lambda x: datetime.strptime(x.findtext("pubDate"), "%a, %d %b %Y %H:%M:%S GMT"), reverse=True)

rss = ET.Element("rss")
channel_out = ET.SubElement(rss, "channel")
for item in selected[:200]:
    channel_out.append(item)

ET.ElementTree(rss).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
json.dump(last_seen, open(LAST_SEEN_FILE, "w"))
