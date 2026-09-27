"""News MCP - News feeds and information."""

from __future__ import annotations
import logging
import requests
from typing import Any, Dict, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.news")


class NewsMCP(MCPBase):
    """News and information MCP module."""

    name = "news"
    description = "News feeds and information"
    commands = [
        "news",
        "headlines",
        "what's happening",
        "latest news",
        "tech news",
        "sports news",
    ]

    def __init__(self):
        super().__init__()
        self._api_key = None  # Optional NewsAPI key
        self._categories = ["general", "technology", "sports", "business", "entertainment"]

    def initialize(self) -> bool:
        """Initialize news module."""
        self._api_key = None  # Can be set via env var NEWS_API_KEY
        self._initialized = True
        log.info("News MCP initialized.")
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle news commands."""
        if command in ("news", "headlines", "what's happening", "latest news"):
            if args:
                return self.get_news(args)
            return self.get_news()
        
        if command in ("tech news",):
            return self.get_news("technology")
        
        if command in ("sports news",):
            return self.get_news("sports")
        
        return None

    def get_news(self, category: str = "general") -> str:
        """Get top headlines."""
        try:
            rss_urls = {
                "general": "http://feeds.bbci.co.uk/news/rss.xml",
                "technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
                "sports": "http://feeds.bbci.co.uk/sport/rss.xml",
            }
            
            url = rss_urls.get(category, rss_urls["general"])
            response = requests.get(url, timeout=4)
            
            if response.status_code == 200:
                import re
                # Handle CDATA sections
                text = response.text.replace('<![CDATA[', '').replace(']]>', '')
                titles = re.findall(r'<title>(.*?)</title>', text)
                if titles:
                    headlines = titles[1:4]
                    lines = []
                    for i, h in enumerate(headlines, 1):
                        h = h.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        h = re.sub(r'<[^>]+>', '', h)  # Remove HTML tags
                        h = h.encode('ascii', 'ignore').decode('ascii').strip()
                        if h and len(h) > 5:
                            lines.append(f"{i}. {h}")
                    if lines:
                        return "Headlines: " + "; ".join(lines)
            
            return "News service is slow right now, sir."
        except requests.exceptions.Timeout:
            return "News service timed out, sir. Your network may be slow."
        except Exception as e:
            log.error("News error: %s", e)
            return "Unable to fetch news, sir."

    def get_news_by_topic(self, topic: str) -> str:
        """Get news about a specific topic."""
        try:
            # Use Google News RSS
            import re
            url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                titles = re.findall(r'<title>(.*?)</title>', response.text)
                if titles:
                    headlines = titles[1:6]
                    lines = []
                    for i, h in enumerate(headlines, 1):
                        # Clean up HTML entities and non-ASCII
                        h = h.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        h = h.encode('ascii', 'ignore').decode('ascii').strip()
                        if h:
                            lines.append(f"{i}. {h}")
                    return f"News about {topic}: " + " ".join(lines)
            
            return f"I couldn't find news about {topic}, sir."
            
        except Exception as e:
            log.error("News search error: %s", e)
            return "I'm having trouble searching the news, sir."

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "categories": self._categories,
        }
