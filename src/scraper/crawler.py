"""
High-level Reddit crawler for depression-related subreddits
"""

import time
from typing import List, Optional, Dict
from pathlib import Path

from src.scraper.client import RedditClient
from src.scraper.parser import RedditParser
from src.scraper.proxy import ProxyManager
from src.utils.logger import setup_logger

logger = setup_logger("depression_alert.scraper.crawler")


class RedditCrawler:
    """Crawl posts and comments from Reddit subreddits"""
    
    # Target subreddits for depression detection
    TARGET_SUBREDDITS = [
        "depression",
        "anxiety", 
        "SuicideWatch",
        "lonely",
        "offmychest",
        "mentalhealth"
    ]
    
    def __init__(
        self,
        delay: float = 2.0,
        use_proxy: bool = False,
        output_dir: str = "data/raw/reddit"
    ):
        self.client = RedditClient(delay=delay, use_proxy=use_proxy)
        self.parser = RedditParser()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def crawl_subreddit(
        self,
        subreddit: str,
        limit: int = 500,
        sort_type: str = "new",
        include_comments: bool = True
    ) -> List[Dict]:
        """
        Crawl posts from a single subreddit
        
        Args:
            subreddit: Name of subreddit (without r/)
            limit: Max number of posts
            sort_type: hot, new, top, rising
            include_comments: Whether to fetch comments
        """
        logger.info(f"Crawling r/{subreddit} ({sort_type}) - limit {limit}")
        
        all_posts = []
        after = None
        
        while len(all_posts) < limit:
            # Build URL
            url = f"https://www.reddit.com/r/{subreddit}/{sort_type}/.json"
            params = f"?limit={min(100, limit - len(all_posts))}"
            if after:
                params += f"&after={after}"
            
            full_url = url + params
            data = self.client.get_json(full_url)
            
            if not data:
                break
            
            # Parse posts
            posts = self.parser.parse_listing(data)
            if not posts:
                break
            
            # Fetch comments if requested
            if include_comments:
                for post in posts:
                    comments = self.crawl_comments(post['permalink'])
                    post['comments'] = comments
            
            all_posts.extend(posts)
            after = data.get('data', {}).get('after')
            
            logger.info(f"  Progress: {len(all_posts)}/{limit} posts")
            time.sleep(self.client.delay)
        
        # Save to file
        self._save_posts(subreddit, all_posts)
        
        return all_posts
    
    def crawl_comments(self, post_permalink: str, limit: int = 500) -> List[Dict]:
        """Crawl comments from a specific post"""
        url = f"https://www.reddit.com{post_permalink}.json"
        data = self.client.get_json(url)
        
        if not data:
            return []
        
        return self.parser.parse_comments(data, limit=limit)
    
    def crawl_all_targets(
        self,
        limit_per_subreddit: int = 300,
        include_comments: bool = False
    ) -> Dict[str, List[Dict]]:
        """Crawl all depression-related subreddits"""
        results = {}
        
        for subreddit in self.TARGET_SUBREDDITS:
            logger.info(f"\n{'='*50}")
            logger.info(f"Crawling r/{subreddit}")
            logger.info(f"{'='*50}")
            
            posts = self.crawl_subreddit(
                subreddit=subreddit,
                limit=limit_per_subreddit,
                include_comments=include_comments
            )
            
            results[subreddit] = posts
            
            # Delay between subreddits
            if subreddit != self.TARGET_SUBREDDITS[-1]:
                time.sleep(5)
        
        return results
    
    def _save_posts(self, subreddit: str, posts: List[Dict]) -> None:
        """Save crawled data to JSON file"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{subreddit}_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(posts)} posts to {filepath}")