import feedparser
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta, timezone
import json
import os
import re
import random

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "final.xml"
LAST_SEEN_FILE = "last_seen_final.json"
MAX_ARTICLE_AGE_DAYS = 1
SIMILARITY_THRESHOLD = 0.65

# ===== LOAD REFERENCE MODEL =====
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# ===== HELPER FUNCTIONS =====

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_last_seen():
    """Load historical URLs safely; auto-handle empty or corrupt files"""
    if os.path.exists(LAST_SEEN_FILE):
        try:
            with open(LAST_SEEN_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {}  # empty file
                data = json.loads(content)

                cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_ARTICLE_AGE_DAYS)
                cleaned = {}
                for url, ts in data.items():
                    try:
                        dt = datetime.fromisoformat(ts)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt > cutoff:
                            cleaned[url] = ts
                    except Exception:
                        continue
                return cleaned
        except json.JSONDecodeError:
            print("⚠️  Warning: last_seen_final.json corrupted — resetting.")
            return {}
        except Exception as e:
            print(f"⚠️  Could not load {LAST_SEEN_FILE}: {e}")
            return {}
    return {}


def save_last_seen(last_seen):
    with open(LAST_SEEN_FILE, "w") as f:
        json.dump(last_seen, f, indent=2)


def load_reference_titles():
    with open(REFERENCE_FILE, "r") as f:
        return [clean_text(line.strip()) for line in f if line.strip()]


def load_feeds():
    with open(FEEDS_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_pubdate(pubdate):
    try:
        dt = datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def title_similarity(title, reference_titles, ref_embeddings):
    title_clean = clean_text(title)
    if not title_clean:
        return 0
    title_emb = model.encode([title_clean])
    sims = cosine_similarity(title_emb, ref_embeddings)
    return float(max(sims[0]))


# ===== MAIN LOGIC =====

def main():
    print("🔄 Starting final feed filter...")
    feeds = load_feeds()
    reference_titles = load_reference_titles()
    ref_embeddings = model.encode(reference_titles)
    last_seen = load_last_seen()
    new_last_seen = last_seen.copy()

    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "Filtered Final Feed"

    total_added = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_ARTICLE_AGE_DAYS)

    for feed_url in feeds:
        print(f"📡 Fetching: {feed_url}")
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            link = entry.get("link", "")
            title = entry.get("title", "")
            pubdate_str = entry.get("published", "")
            description = entry.get("summary", "")

            if not link or not title:
                continue

            pubdate = parse_pubdate(pubdate_str)
            if pubdate < cutoff:
                continue

            if link in last_seen:
                continue

            sim_score = title_similarity(title, reference_titles, ref_embeddings)
            if sim_score < SIMILARITY_THRESHOLD:
                continue

            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = title
            ET.SubElement(item, "link").text = link
            ET.SubElement(item, "pubDate").text = pubdate.strftime("%a, %d %b %Y %H:%M:%S %Z")
            ET.SubElement(item, "description").text = description

            new_last_seen[link] = datetime.now(timezone.utc).isoformat()
            total_added += 1

    tree = ET.ElementTree(root)
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

    save_last_seen(new_last_seen)

    print(f"✅ Final feed updated: {OUTPUT_FILE}")
    print(f"🆕 Articles added: {total_added}")


if __name__ == "__main__":
    main()
