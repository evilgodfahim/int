import feedparser
import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime, timezone, timedelta
import sys

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
OUTPUT_FILE = "temp.xml"
LAST_SEEN_FILE = "last_seen_temp.json"
MAX_ARTICLE_AGE_HOURS = 24  # Keep only articles from last 24 hours

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ===== SOURCE DETECTION =====
def get_source(entry):
    """Identify news source from URL"""
    url = entry.get("link", "").lower()
    source_map = {
        "bbc": "BBC",
        "nytimes": "The New York Times",
        "aljazeera": "Al Jazeera",
        "scmp": "South China Morning Post",
        "hindu": "The Hindu",
        "asiatimes": "Asia Times",
        "eurasiareview": "Eurasia Review",
        "middleeasteye": "Middle East Eye",
        "middleeastmonitor": "Middle East Monitor",
        "moscowtimes": "The Moscow Times",
        "financialexpress": "Financial Express",
        "tbsnews": "The Business Standard",
        "thedailystar": "The Daily Star",
        "unb": "United News Bangladesh"
    }
    for key, name in source_map.items():
        if key in url:
            return name
    return "Unknown"

# ===== DATE PARSING =====
def parse_date(entry):
    """Parse publication date from feed entry"""
    if "published_parsed" in entry and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except:
            pass
    return datetime.now(timezone.utc)

def is_recent(pub_date):
    """Check if article is within the 24-hour window"""
    now = datetime.now(timezone.utc)
    age = now - pub_date
    return age < timedelta(hours=MAX_ARTICLE_AGE_HOURS)

# ===== LAST SEEN MANAGEMENT =====
def load_last_seen():
    """Load URL tracking to prevent duplicates within 24 hours"""
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, "r") as f:
            data = json.load(f)
            # Clean old entries (>24 hours)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_ARTICLE_AGE_HOURS)
            return {url: ts for url, ts in data.items() 
                   if datetime.fromisoformat(ts) > cutoff}
    return {}

def save_last_seen(data):
    """Save URL tracking"""
    with open(LAST_SEEN_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ===== XML MANAGEMENT =====
def load_existing_xml():
    """Load existing temp.xml or create new structure"""
    if os.path.exists(OUTPUT_FILE):
        tree = ET.parse(OUTPUT_FILE)
        root = tree.getroot()
        return tree, root
    else:
        root = ET.Element("rss", version="2.0")
        channel = ET.SubElement(root, "channel")
        ET.SubElement(channel, "title").text = "Temporary News Collection"
        ET.SubElement(channel, "link").text = "https://evilgodfahim.github.io/"
        ET.SubElement(channel, "description").text = "24-hour rolling news window"
        return ET.ElementTree(root), root

def clean_old_articles(root):
    """Remove articles older than 24 hours from XML"""
    channel = root.find("channel")
    if channel is None:
        return
    
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_ARTICLE_AGE_HOURS)
    items_to_remove = []
    
    for item in channel.findall("item"):
        pub_date_str = item.findtext("pubDate", "")
        try:
            pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
            pub_date = pub_date.replace(tzinfo=timezone.utc)
            if pub_date < cutoff:
                items_to_remove.append(item)
        except:
            # If can't parse date, remove it to be safe
            items_to_remove.append(item)
    
    for item in items_to_remove:
        channel.remove(item)
    
    return len(items_to_remove)

# ===== MAIN COLLECTION LOGIC =====
def collect_articles():
    """Main function: collect new articles from all feeds"""
    
    # Load feeds list
    if not os.path.exists(FEEDS_FILE):
        print(f"❌ {FEEDS_FILE} not found")
        return
    
    with open(FEEDS_FILE, "r") as f:
        feed_urls = [line.strip() for line in f 
                    if line.strip() and not line.startswith("#")]
    
    print(f"📡 Fetching from {len(feed_urls)} feeds...")
    
    # Load tracking data
    last_seen = load_last_seen()
    tree, root = load_existing_xml()
    
    # Clean old articles first
    removed_count = clean_old_articles(root)
    print(f"🗑️  Removed {removed_count} old articles (>24h)")
    
    # Collect new articles
    new_articles = []
    feed_errors = []
    
    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                # Basic validation
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                
                if not title or not link:
                    continue
                
                # Skip if already seen in last 24 hours
                if link in last_seen:
                    continue
                
                # Parse date and check recency
                pub_date = parse_date(entry)
                if not is_recent(pub_date):
                    continue
                
                # Get source
                source = get_source(entry)
                
                # Add to collection
                new_articles.append({
                    "title": title,
                    "link": link,
                    "source": source,
                    "pubDate": pub_date
                })
                
                # Mark as seen
                last_seen[link] = pub_date.isoformat()
        
        except Exception as e:
            feed_errors.append(f"{feed_url}: {str(e)}")
    
    # Add new articles to XML
    channel = root.find("channel")
    for article in new_articles:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = article["title"]
        ET.SubElement(item, "link").text = article["link"]
        ET.SubElement(item, "source").text = article["source"]
        ET.SubElement(item, "pubDate").text = article["pubDate"].strftime("%a, %d %b %Y %H:%M:%S GMT")
    
    # Update channel metadata
    last_build = channel.find("lastBuildDate")
    if last_build is None:
        last_build = ET.SubElement(channel, "lastBuildDate")
    last_build.text = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    
    # Save everything
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    save_last_seen(last_seen)
    
    # Report
    total_items = len(channel.findall("item"))
    print(f"✅ Added {len(new_articles)} new articles")
    print(f"📦 Total in temp.xml: {total_items} articles")
    print(f"⏱️  Processing time: ~{len(feed_urls) * 0.5:.1f} seconds")
    
    if feed_errors:
        print(f"⚠️  Feed errors ({len(feed_errors)}):")
        for error in feed_errors[:5]:  # Show first 5
            print(f"   {error}")
    
    # Exit with success
    sys.exit(0)

if __name__ == "__main__":
    try:
        collect_articles()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
