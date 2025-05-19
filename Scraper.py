import feedparser

def fetch_latest_news():
    """Fetches recent news headlines from Google News RSS."""
    url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    headlines = []
    for entry in feed.entries[:10]:
        headlines.append({
            "title": entry.title,
            "link": entry.link
        })
    return headlines
