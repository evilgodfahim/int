import feedparser
import hashlib
import os
from datetime import datetime
import xml.etree.ElementTree as ET

# ✅ All 36 feeds included
FEEDS = [
    "http://feeds.feedburner.com/dawn-news-world",
    "https://feeds.guardian.co.uk/theguardian/world/rss",
    "https://www.ft.com/rss/world",
    "https://feeds.feedburner.com/AtlanticInternational",
    "https://theconversation.com/articles.atom",
    "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
    "https://politepol.com/fd/IleailW8Do7p.xml",
    "https://www.ft.com/rss/world/brussels",
    "https://www.ft.com/rss/companies/energy",
    "https://www.ft.com/rss/home/asia",
    "https://www.themoscowtimes.com/rss/news",
    "https://politepol.com/fd/x7ZadWalRg3O.xml",
    "https://politepol.com/fd/2wwElTcUpcfo.xml",
    "https://politepol.com/fd/FQtkYoIiTwrT.xml",
    "https://politepol.com/fd/MrLghs9CUuhs.xml",
    "https://politepol.com/fd/nTjHEYiFFmDe.xml",
    "https://www.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.aljazeera.com/Services/Rss/?PostingId=2007731105943979989",
    "https://www.scmp.com/rss/5/feed",
    "https://www.scmp.com/rss/318199/feed",  # ✅ newly added SCMP feed
    "https://www.indiatoday.in/rss/1206577",
    "https://www.thehindu.com/news/international/?service=rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://politepol.com/fd/IZHDnFfjhvdc.xml",
    "https://politepol.com/fd/iFF48wN05TWX.xml",
    "https://politepol.com/fd/hKA2ZijBkurO.xml",
    "https://politepol.com/fd/ejjxAclQ0Ij0.xml",
    "https://politepol.com/fd/mxmm1zHl3Vkp.xml",
    "https://politepol.com/fd/eU6sPuvezFmi.xml",
    "https://politepol.com/fd/mgL5tJODdXMU.xml",
    "https://politepol.com/fd/jxdB7qx3NSFJ.xml",
    "https://politepol.com/fd/aGCRq4aQKpwL.xml",
    "https://politepol.com/fd/twCQYifhldi8.xml",
    "https://politepol.com/fd/RVHJinKtHIEp.xml",
    "https://politepol.com/fd/MQdEEfACJVgu.xml"
]

OUTPUT_FILE = "combined.xml"
INDEX_FILE = "index.txt"
MAX_ARTICLES = 500  # keep last 500

def get_id(entry):
    """Generate unique ID for each entry"""
    base = entry.get("id") or entry.get("link") or entry.get("title", "")
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def load_seen():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()

def save_seen(seen):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        for item in seen:
            f.write(item + "\n")

def main():
    seen = load_seen()
    new_articles = []

    print(f"Processing {len(FEEDS)} feeds...")  # ✅ feed count check

    for feed in FEEDS:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries:
            uid = get_id(entry)
            if uid not in seen:
                seen.add(uid)
                new_articles.append((entry, uid))

    # Sort newest first (by published date if available)
    def sort_key(item):
        entry = item[0]
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        return datetime.now()
    new_articles.sort(key=sort_key, reverse=True)

    # Trim to max limit
    all_articles = new_articles[:MAX_ARTICLES]

    # Save updated seen list (only keep 500)
    save_seen(set(uid for _, uid in all_articles))

    # Build RSS XML
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Combined Feed"
    ET.SubElement(channel, "link").text = "https://yourusername.github.io/combined.xml"
    ET.SubElement(channel, "description").text = "Aggregated feed without duplicates"
    ET.SubElement(channel, "lastBuildDate").text = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

    for entry, uid in all_articles:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = entry.get("title", "No title")
        ET.SubElement(item, "link").text = entry.get("link", "")
        ET.SubElement(item, "guid").text = uid
        ET.SubElement(item, "description").text = entry.get("summary", "")
        pub_date = entry.get("published", datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"))
        ET.SubElement(item, "pubDate").text = pub_date

    tree = ET.ElementTree(rss)
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    main() 
