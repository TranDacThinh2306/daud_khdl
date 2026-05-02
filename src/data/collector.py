"""
collector.py - Thu thập comment từ social media APIs
=====================================================
Collects comments from social media platforms (Reddit, Twitter)
for depression detection analysis.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import requests
import json
import time
import random
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)



class SocialMediaCollector:
    """Collects social media comments from various platforms."""

    def __init__(self, platform: str = "reddit", config: Optional[Dict] = None):
        """
        Initialize the collector.

        Args:
            platform: Social media platform ('reddit', 'twitter')
            config: API credentials and configuration
        """
        self.platform = platform.lower()
        self.config = config or {}
        self._client = None

    def _init_reddit_client(self):
        """Initialize Reddit API client using PRAW."""
        try:
            import praw

            self._client = praw.Reddit(
                client_id=self.config.get("client_id", ""),
                client_secret=self.config.get("client_secret", ""),
                user_agent=self.config.get("user_agent", "depression_xai_collector/1.0"),
            )
            logger.info("Reddit client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise

    def _init_twitter_client(self):
        """Initialize Twitter API client using Tweepy."""
        try:
            import tweepy

            auth = tweepy.OAuthHandler(
                self.config.get("api_key", ""),
                self.config.get("api_secret", ""),
            )
            auth.set_access_token(
                self.config.get("access_token", ""),
                self.config.get("access_token_secret", ""),
            )
            self._client = tweepy.API(auth, wait_on_rate_limit=True)
            logger.info("Twitter client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            raise

    def connect(self):
        """Connect to the configured social media platform."""
        if self.platform == "reddit":
            self._init_reddit_client()
        elif self.platform == "twitter":
            self._init_twitter_client()
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")

    def collect_reddit_comments(
        self,
        subreddits: List[str],
        limit: int = 1000,
        keywords: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Collect comments from specified subreddits.

        Args:
            subreddits: List of subreddit names (e.g., ['depression', 'mentalhealth'])
            limit: Maximum number of comments per subreddit
            keywords: Optional keyword filter

        Returns:
            DataFrame with collected comments
        """
        if self._client is None:
            self.connect()

        comments = []
        for subreddit_name in subreddits:
            logger.info(f"Collecting from r/{subreddit_name}...")
            try:
                subreddit = self._client.subreddit(subreddit_name)
                for submission in subreddit.hot(limit=limit):
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        text = comment.body.strip()
                        if keywords and not any(kw in text.lower() for kw in keywords):
                            continue
                        comments.append({
                            "comment_id": comment.id,
                            "user_id": str(comment.author) if comment.author else "deleted",
                            "text": text,
                            "platform": "reddit",
                            "subreddit": subreddit_name,
                            "timestamp": datetime.fromtimestamp(comment.created_utc).isoformat(),
                            "score": comment.score,
                        })
            except Exception as e:
                logger.warning(f"Error collecting from r/{subreddit_name}: {e}")

        df = pd.DataFrame(comments)
        logger.info(f"Collected {len(df)} comments from Reddit")
        return df

    def collect_twitter_posts(
        self,
        query: str,
        limit: int = 1000,
        lang: str = "en",
    ) -> pd.DataFrame:
        """
        Collect tweets matching a query.

        Args:
            query: Search query string
            limit: Maximum number of tweets
            lang: Language filter

        Returns:
            DataFrame with collected tweets
        """
        if self._client is None:
            self.connect()

        tweets = []
        try:
            for tweet in self._client.search_tweets(
                q=query, lang=lang, count=min(limit, 100), tweet_mode="extended"
            ):
                tweets.append({
                    "comment_id": tweet.id_str,
                    "user_id": tweet.user.id_str,
                    "text": tweet.full_text,
                    "platform": "twitter",
                    "timestamp": tweet.created_at.isoformat(),
                })
        except Exception as e:
            logger.error(f"Error collecting tweets: {e}")

        df = pd.DataFrame(tweets)
        logger.info(f"Collected {len(df)} tweets")
        return df

    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """Save collected data to CSV."""
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved {len(df)} records to {output_path}")

import requests
import json
import pandas as pd
from datetime import datetime
import time
import random
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================
# ENUM VÀ DATACLASS
# ============================================

class SortType(Enum):
    """Các loại sắp xếp bài viết"""
    HOT = "hot"
    NEW = "new"
    TOP = "top"
    RISING = "rising"
    CONTROVERSIAL = "controversial"

class TimeFilter(Enum):
    """Bộ lọc thời gian cho top/controversial"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL = "all"

@dataclass
class RedditPost:
    """Data class cho bài viết Reddit"""
    id: str
    title: str
    content: str
    subreddit: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    created_date: str
    url: str
    is_self: bool
    flair: Optional[str]
    over_18: bool
    
    def to_dict(self) -> Dict:
        """Chuyển đổi thành dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Chuyển đổi thành JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

@dataclass
class RedditComment:
    """Data class cho bình luận Reddit"""
    id: str
    post_id: str
    author: str
    body: str
    score: int
    created_utc: float
    created_date: str
    parent_id: str
    depth: int
    is_submitter: bool
    
    def to_dict(self) -> Dict:
        """Chuyển đổi thành dictionary"""
        return asdict(self)

# ============================================
# LỚP QUẢN LÝ PROXY
# ============================================

class ProxyManager:
    """Quản lý proxy xoay vòng"""
    
    def __init__(self, proxy_list: Optional[List[str]] = None):
        """
        Khởi tạo ProxyManager
        
        Args:
            proxy_list: Danh sách proxy (định dạng: http://ip:port hoặc https://ip:port)
        """
        self.proxy_list = proxy_list or []
        self.current_index = 0
        self.failed_proxies = set()
    
    def add_proxy(self, proxy: str) -> None:
        """Thêm proxy mới"""
        if proxy not in self.proxy_list:
            self.proxy_list.append(proxy)
    
    def add_proxies(self, proxies: List[str]) -> None:
        """Thêm nhiều proxy cùng lúc"""
        for proxy in proxies:
            self.add_proxy(proxy)
    
    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """
        Lấy proxy tiếp theo trong danh sách (xoay vòng)
        
        Returns:
            Dict định dạng {'http': proxy, 'https': proxy} hoặc None nếu không có proxy
        """
        if not self.proxy_list:
            return None
        
        # Tìm proxy chưa bị fail
        attempts = 0
        while attempts < len(self.proxy_list):
            proxy = self.proxy_list[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxy_list)
            
            if proxy not in self.failed_proxies:
                return {'http': proxy, 'https': proxy}
            attempts += 1
        
        # Tất cả proxy đều đã fail
        print("⚠️ Tất cả proxy đều đã thất bại")
        return None
    
    def mark_proxy_failed(self, proxy: str) -> None:
        """Đánh dấu proxy đã thất bại"""
        self.failed_proxies.add(proxy)
    
    def reset_failed_proxies(self) -> None:
        """Reset danh sách proxy đã fail"""
        self.failed_proxies.clear()
    
    def get_free_proxies_from_api(self) -> List[str]:
        """
        Lấy danh sách proxy miễn phí từ API công khai
        
        Returns:
            List[str]: Danh sách proxy (định dạng http://ip:port)
        """
        try:
            # Từ ProxyRoller
            url = "https://api.proxyroll.com/v2/?get=true&https=true&anonymity=true"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                proxies = response.text.strip().split('\n')
                return [f"http://{p}" for p in proxies if p]
        except Exception as e:
            print(f"⚠️ Không thể lấy proxy từ API: {e}")
        
        return []

# ============================================
# LỚP XỬ LÝ REQUEST
# ============================================

class RedditRequestHandler:
    """Xử lý request tới Reddit API"""
    
    DEFAULT_USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ]
    
    def __init__(
        self, 
        delay: float = 2.0,
        max_retries: int = 3,
        use_proxy: bool = False,
        proxy_manager: Optional[ProxyManager] = None
    ):
        """
        Khởi tạo RedditRequestHandler
        
        Args:
            delay: Thời gian delay giữa các request (giây)
            max_retries: Số lần thử lại tối đa khi request thất bại
            use_proxy: Có sử dụng proxy không
            proxy_manager: ProxyManager instance (tự tạo nếu None)
        """
        self.delay = delay
        self.max_retries = max_retries
        self.use_proxy = use_proxy
        self.proxy_manager = proxy_manager or ProxyManager()
        self.last_request_time = 0
    
    def _get_headers(self) -> Dict[str, str]:
        """Tạo headers cho request"""
        return {
            "User-Agent": random.choice(self.DEFAULT_USER_AGENTS),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.reddit.com/",
            "Connection": "keep-alive"
        }
    
    def _respect_rate_limit(self) -> None:
        """Đảm bảo không gửi request quá nhanh"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def get_json(self, url: str) -> Optional[Dict]:
        """
        Gửi GET request và trả về JSON
        
        Args:
            url: URL cần request
            
        Returns:
            JSON data hoặc None nếu thất bại
        """
        self._respect_rate_limit()
        
        for attempt in range(self.max_retries):
            headers = self._get_headers()
            proxies = None
            
            if self.use_proxy:
                proxies = self.proxy_manager.get_next_proxy()
                if proxies:
                    print(f"  🔄 Sử dụng proxy: {proxies['http']}")
            
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=30,
                    allow_redirects=True
                )
                
                # Xử lý các status code
                if response.status_code == 200:
                    return response.json()
                    
                elif response.status_code == 429:
                    print(f"  ⚠️ Rate limit! (Attempt {attempt + 1}/{self.max_retries})")
                    wait_time = (attempt + 1) * 30
                    print(f"  ⏸️  Đợi {wait_time} giây...")
                    time.sleep(wait_time)
                    
                elif response.status_code == 403:
                    print(f"  ❌ Bị chặn (403) - Thử đổi User-Agent")
                    # Xoay User-Agent mới
                    continue
                    
                else:
                    print(f"  ❌ HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"  ⏱️ Timeout (Attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.ConnectionError:
                print(f"  🔌 Connection error (Attempt {attempt + 1}/{self.max_retries})")
                
                if proxies and self.use_proxy:
                    proxy_url = proxies['http']
                    self.proxy_manager.mark_proxy_failed(proxy_url)
                    print(f"  🚫 Đã đánh dấu proxy {proxy_url} là failed")
                    
            except Exception as e:
                print(f"  ❌ Lỗi: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.delay * (attempt + 1))
        
        return None

# ============================================
# LỚP PARSER CHÍNH
# ============================================

class RedditParser:
    """Parse dữ liệu JSON từ Reddit"""
    
    @staticmethod
    def parse_post(json_data: Dict) -> Optional[RedditPost]:
        """
        Parse bài viết từ JSON
        
        Args:
            json_data: JSON từ Reddit API
            
        Returns:
            RedditPost object hoặc None
        """
        if not json_data or len(json_data) == 0:
            return None
        
        try:
            post_listing = json_data[0]
            post_data = post_listing["data"]["children"][0]["data"]
            
            return RedditPost(
                id=post_data.get("id", ""),
                title=post_data.get("title", ""),
                content=post_data.get("selftext", "")[:1000],
                subreddit=post_data.get("subreddit", ""),
                author=post_data.get("author", "[deleted]"),
                score=post_data.get("score", 0),
                upvote_ratio=post_data.get("upvote_ratio", 0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=post_data.get("created_utc", 0),
                created_date=datetime.fromtimestamp(
                    post_data.get("created_utc", 0)
                ).strftime("%Y-%m-%d %H:%M:%S"),
                url=f"https://reddit.com{post_data.get('permalink', '')}",
                is_self=post_data.get("is_self", False),
                flair=post_data.get("link_flair_text"),
                over_18=post_data.get("over_18", False)
            )
        except (KeyError, IndexError, TypeError) as e:
            print(f"❌ Lỗi parse post: {e}")
            return None
    
    @staticmethod
    def parse_comments(json_data: Dict, post_id: str, max_comments: int = 500) -> List[RedditComment]:
        """
        Parse comments từ JSON (bao gồm nested comments)
        
        Args:
            json_data: JSON từ Reddit API
            post_id: ID của bài viết
            max_comments: Số lượng comments tối đa
            
        Returns:
            List[RedditComment]: Danh sách comments
        """
        comments = []
        
        if not json_data or len(json_data) < 2:
            return comments
        
        try:
            comments_listing = json_data[1]
            comments_raw = comments_listing["data"]["children"]
            
            for comment_node in comments_raw:
                if len(comments) >= max_comments:
                    break
                
                comment_data = comment_node.get("data", {})
                
                # Bỏ qua MoreComments placeholder
                if comment_node.get("kind") == "more":
                    continue
                
                # Bỏ qua comment đã xóa
                if comment_data.get("body") in ["[deleted]", "[removed]"]:
                    continue
                if comment_data.get("author") in ["[deleted]", None]:
                    continue
                
                comment = RedditComment(
                    id=comment_data.get("id", ""),
                    post_id=post_id,
                    author=comment_data.get("author", "unknown"),
                    body=comment_data.get("body", "")[:500],
                    score=comment_data.get("score", 0),
                    created_utc=comment_data.get("created_utc", 0),
                    created_date=datetime.fromtimestamp(
                        comment_data.get("created_utc", 0)
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    parent_id=comment_data.get("parent_id", ""),
                    depth=comment_data.get("depth", 0),
                    is_submitter=comment_data.get("is_submitter", False)
                )
                comments.append(comment)
                
        except (KeyError, TypeError) as e:
            print(f"⚠️ Lỗi parse comments: {e}")
        
        return comments

# ============================================
# LỚP QUẢN LÝ SUBREDDIT
# ============================================

class SubredditCrawler:
    """Crawl dữ liệu từ một subreddit cụ thể"""
    
    def __init__(
        self,
        name: str,
        request_handler: RedditRequestHandler,
        sort_type: SortType = SortType.NEW,
        time_filter: TimeFilter = TimeFilter.ALL,
        limit: int = 100
    ):
        """
        Khởi tạo SubredditCrawler
        
        Args:
            name: Tên subreddit (không bao gồm r/)
            request_handler: RedditRequestHandler instance
            sort_type: Loại sắp xếp
            time_filter: Bộ lọc thời gian
            limit: Số lượng bài viết tối đa
        """
        self.name = name
        self.request_handler = request_handler
        self.sort_type = sort_type
        self.time_filter = time_filter
        self.limit = limit
    
    def _build_url(self, after: Optional[str] = None) -> str:
        """Xây dựng URL cho subreddit"""
        base_url = f"https://www.reddit.com/r/{self.name}/{self.sort_type.value}/.json"
        params = f"?limit={min(100, self.limit)}"
        
        if self.sort_type in [SortType.TOP, SortType.CONTROVERSIAL]:
            params += f"&t={self.time_filter.value}"
        
        if after:
            params += f"&after={after}"
        
        return base_url + params
    
    def get_posts(self) -> List[RedditPost]:
        """
        Lấy danh sách bài viết từ subreddit
        
        Returns:
            List[RedditPost]: Danh sách bài viết
        """
        posts = []
        after = None
        
        print(f"\n📁 Đang crawl r/{self.name} ({self.sort_type.value})")
        
        while len(posts) < self.limit:
            url = self._build_url(after)
            json_data = self.request_handler.get_json(url)
            
            if not json_data:
                break
            
            try:
                # Parse bài viết từ response
                post_listing = json_data["data"]
                new_posts = post_listing["children"]
                
                if not new_posts:
                    break
                
                for post_node in new_posts:
                    post_data = post_node["data"]
                    post = RedditParser.parse_post([{"data": {"children": [post_node]}}])
                    if post:
                        posts.append(post)
                
                # Lấy token cho trang tiếp theo
                after = post_listing.get("after")
                if not after:
                    break
                
                print(f"  ✅ Đã lấy {len(posts)}/{self.limit} bài viết")
                
            except KeyError as e:
                print(f"  ❌ Lỗi parse JSON: {e}")
                break
        
        print(f"  📊 Hoàn thành: {len(posts)} bài viết từ r/{self.name}")
        return posts

# ============================================
# LỚP CHÍNH - REDDIT SCRAPER
# ============================================

class RedditScraper:
    """Lớp chính điều khiển toàn bộ quá trình crawl"""
    
    def __init__(
        self,
        delay: float = 2.0,
        max_retries: int = 3,
        use_proxy: bool = False,
        output_dir: str = "reddit_data"
    ):
        """
        Khởi tạo RedditScraper
        
        Args:
            delay: Delay giữa các request (giây)
            max_retries: Số lần thử lại tối đa
            use_proxy: Có sử dụng proxy không
            output_dir: Thư mục lưu dữ liệu
        """
        self.proxy_manager = ProxyManager() if use_proxy else None
        self.request_handler = RedditRequestHandler(
            delay=delay,
            max_retries=max_retries,
            use_proxy=use_proxy,
            proxy_manager=self.proxy_manager
        )
        self.storage = DataStorage(output_dir)
        self.parser = RedditParser()
    
    def fetch_post_by_url(self, url: str, include_comments: bool = True) -> Optional[RedditPost]:
        """
        Crawl một bài viết cụ thể từ URL
        
        Args:
            url: URL bài viết Reddit (có thể có hoặc không .json)
            include_comments: Có lấy comments không
            
        Returns:
            RedditPost hoặc None
        """
        # Đảm bảo URL kết thúc bằng .json
        if not url.endswith('.json'):
            url = url.rstrip('/') + '.json'
        
        print(f"\n{'='*60}")
        print(f"🎯 Crawl bài viết: {url}")
        print(f"{'='*60}")
        
        # Lấy JSON
        json_data = self.request_handler.get_json(url)
        
        if not json_data:
            print("❌ Không thể lấy dữ liệu")
            return None
        
        # Parse bài viết
        post = self.parser.parse_post(json_data)
        
        if not post:
            print("❌ Không thể parse bài viết")
            return None
        
        print(f"✅ Đã lấy bài viết: {post.title[:60]}...")
        
        # Lưu bài viết
        self.storage.save_posts([post], post.subreddit)
        
        # Lấy comments nếu cần
        if include_comments and post.num_comments > 0:
            print(f"\n💬 Đang lấy {min(500, post.num_comments)} comments...")
            comments = self.parser.parse_comments(json_data, post.id, max_comments=500)
            if comments:
                self.storage.save_comments(comments, post.id)
        
        return post
    
    def fetch_subreddit(
        self,
        subreddit_name: str,
        sort_type: SortType = SortType.NEW,
        time_filter: TimeFilter = TimeFilter.ALL,
        limit: int = 100,
        include_comments: bool = False
    ) -> List[RedditPost]:
        """
        Crawl nhiều bài viết từ một subreddit
        
        Args:
            subreddit_name: Tên subreddit
            sort_type: Loại sắp xếp
            time_filter: Bộ lọc thời gian
            limit: Số lượng bài viết tối đa
            include_comments: Có lấy comments cho mỗi bài không
            
        Returns:
            List[RedditPost]: Danh sách bài viết
        """
        print(f"\n{'='*60}")
        print(f"📁 Crawl subreddit: r/{subreddit_name}")
        print(f"   Sort: {sort_type.value} | Limit: {limit}")
        print(f"{'='*60}")
        
        # Tạo crawler cho subreddit
        subreddit_crawler = SubredditCrawler(
            name=subreddit_name,
            request_handler=self.request_handler,
            sort_type=sort_type,
            time_filter=time_filter,
            limit=limit
        )
        
        # Lấy bài viết
        posts = subreddit_crawler.get_posts()
        
        if not posts:
            print("❌ Không lấy được bài viết nào")
            return []
        
        # Lưu bài viết
        self.storage.save_posts(posts, subreddit_name)
        
        # Lấy comments nếu cần (chỉ cho 20 bài đầu để tránh quá tải)
        if include_comments:
            print(f"\n💬 Đang lấy comments cho {min(20, len(posts))} bài đầu tiên...")
            for i, post in enumerate(posts[:20]):
                print(f"  📝 Bài {i+1}: {post.title[:40]}...")
                post_url = post.url + '.json'
                json_data = self.request_handler.get_json(post_url)
                if json_data:
                    comments = self.parser.parse_comments(json_data, post.id, max_comments=200)
                    if comments:
                        self.storage.save_comments(comments, post.id)
                time.sleep(self.request_handler.delay)
        
        return posts
    
    def fetch_multiple_subreddits(
        self,
        subreddits: List[str],
        sort_type: SortType = SortType.NEW,
        limit_per_subreddit: int = 50,
        include_comments: bool = False
    ) -> Dict[str, List[RedditPost]]:
        """
        Crawl nhiều subreddit cùng lúc
        
        Args:
            subreddits: Danh sách tên subreddit
            sort_type: Loại sắp xếp
            limit_per_subreddit: Số bài viết tối đa mỗi subreddit
            include_comments: Có lấy comments không
            
        Returns:
            Dict: {subreddit_name: list_of_posts}
        """
        results = {}
        
        for subreddit in subreddits:
            posts = self.fetch_subreddit(
                subreddit_name=subreddit,
                sort_type=sort_type,
                limit=limit_per_subreddit,
                include_comments=include_comments
            )
            results[subreddit] = posts
            
            # Nghỉ giữa các subreddit
            if subreddit != subreddits[-1]:
                print(f"\n⏸️  Nghỉ 5 giây trước khi sang subreddit tiếp theo...")
                time.sleep(5)
        
        return results
    
    def get_statistics(self, posts: List[RedditPost]) -> Dict:
        """Thống kê dữ liệu bài viết"""
        if not posts:
            return {}
        
        df = pd.DataFrame([p.to_dict() for p in posts])
        
        stats = {
            "total_posts": len(posts),
            "total_comments": sum(p.num_comments for p in posts),
            "avg_score": df['score'].mean(),
            "avg_upvote_ratio": df['upvote_ratio'].mean(),
            "subreddits": df['subreddit'].value_counts().to_dict(),
            "top_5_posts": df.nlargest(5, 'score')[['title', 'score']].to_dict('records')
        }
        
        return stats

# ============================================
# VÍ DỤ SỬ DỤNG
# ============================================

def main():
    """Hàm chính demo cách sử dụng"""
    
    print("=" * 60)
    print("🐍 REDDIT SCRAPER - OOP DESIGN")
    print("=" * 60)
    
    # Khởi tạo scraper
    scraper = RedditScraper(
        delay=2.0,          # Delay 2 giây giữa các request
        max_retries=3,      # Thử lại tối đa 3 lần
        use_proxy=False,    # Không dùng proxy (có thể bật lên nếu cần)
        output_dir="my_reddit_data"
    )
    
    # ===== VÍ DỤ 1: Crawl một bài viết cụ thể =====
    print("\n" + "="*60)
    print("VÍ DỤ 1: Crawl một bài viết cụ thể")
    print("="*60)
    
    post_url = "https://www.reddit.com/r/webscraping/comments/1t080rn/how_to_scrape_reddit_now_closed_api"
    post = scraper.fetch_post_by_url(post_url, include_comments=True)
    
    # ===== VÍ DỤ 2: Crawl một subreddit =====
    print("\n" + "="*60)
    print("VÍ DỤ 2: Crawl subreddit")
    print("="*60)
    
    posts = scraper.fetch_subreddit(
        subreddit_name="python",
        sort_type=SortType.HOT,
        limit=20,               # Chỉ lấy 20 bài để demo
        include_comments=False
    )
    
    # Thống kê
    stats = scraper.get_statistics(posts)
    print("\n📊 THỐNG KÊ:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # ===== VÍ DỤ 3: Crawl nhiều subreddit về sức khỏe tâm thần =====
    print("\n" + "="*60)
    print("VÍ DỤ 3: Crawl các subreddit về sức khỏe tâm thần")
    print("="*60)
    
    mental_health_subreddits = [
        "depression",
        "anxiety",
        "mentalhealth"
    ]
    
    results = scraper.fetch_multiple_subreddits(
        subreddits=mental_health_subreddits,
        sort_type=SortType.NEW,
        limit_per_subreddit=30,
        include_comments=False
    )
    
    # Tổng kết
    print("\n" + "="*60)
    print("📊 TỔNG KẾT")
    print("="*60)
    
    for subreddit, posts_list in results.items():
        print(f"   r/{subreddit}: {len(posts_list)} bài viết")
    
    print("\n✅ Hoàn thành!")

# ============================================
# CHẠY CHƯƠNG TRÌNH
# ============================================
if __name__ == "__main__":
    main()