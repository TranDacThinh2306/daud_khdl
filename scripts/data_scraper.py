#!/usr/bin/env python3
"""
Reddit Scraper với Proxy Rotation
- Tự động xoay proxy mỗi N request
- Fallback khi proxy chết
- Lưu dữ liệu theo checkpoint
- Hỗ trợ crawl nhiều subreddit về sức khỏe tâm thần
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.client import RedditClient
from src.utils.logger import setup_logger

# Khởi tạo logger
logger = setup_logger("depression_alert.scraper.crawler")


@dataclass
class CrawlStats:
    """Thống kê quá trình crawl"""
    start_time: float = 0
    end_time: float = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_posts: int = 0
    proxies_used: Dict[str, int] = field(default_factory=dict)
    subreddit_stats: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def elapsed_seconds(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict:
        """Chuyển thành dictionary để lưu"""
        return {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'elapsed_seconds': self.elapsed_seconds,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'total_posts': self.total_posts,
            'proxies_used': self.proxies_used,
            'subreddit_stats': self.subreddit_stats
        }


class RedditCrawlerWithRotation:
    """Reddit Crawler với proxy rotation tự động"""
    
    # Target subreddits cho depression detection
    DEFAULT_SUBREDDITS = [
        "depression",
        "anxiety", 
        "SuicideWatch",
        "lonely",
        "offmychest",
        "mentalhealth"
    ]
    
    def __init__(
        self,
        proxy_rotation_interval: int = 100,
        delay_between_requests: float = 2.0,
        output_dir: str = "data/raw/reddit",
        checkpoint_interval: int = 50,
        max_posts_per_subreddit: int = 300,
        use_proxy: bool = True,
        max_retries: int = 3,
        proxy_config: Optional[Dict] = None
    ):
        """
        Args:
            proxy_rotation_interval: Số request trước khi xoay proxy
            delay_between_requests: Delay giữa các request (giây)
            output_dir: Thư mục lưu dữ liệu
            checkpoint_interval: Lưu checkpoint sau mỗi N bài
            max_posts_per_subreddit: Số bài tối đa mỗi subreddit
            use_proxy: Có sử dụng proxy không
            max_retries: Số lần thử lại tối đa
            proxy_config: Cấu hình cho ProxyManager
        """
        self.rotation_interval = proxy_rotation_interval
        self.delay = delay_between_requests
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_posts_per_subreddit = max_posts_per_subreddit
        
        # Tạo thư mục output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cấu hình proxy mặc định
        default_proxy_config = {
            'auto_fetch_free': True,
            'check_on_start': False,
            'lazy_check': True,
            'max_failures': 3,
            'rotation_strategy': 'fastest',
            'check_timeout': 10,
            'max_check_workers': 10,
            'min_proxies': 3,
            'refresh_interval_minutes': 30
        }
        
        if proxy_config:
            default_proxy_config.update(proxy_config)
        
        # Khởi tạo Reddit client
        self.client = RedditClient(
            delay=delay_between_requests,
            use_proxy=use_proxy,
            max_retries=max_retries,
            proxy_rotation_interval=proxy_rotation_interval,
            proxy_config=default_proxy_config if use_proxy else None
        )
        
        # Thống kê
        self.stats = CrawlStats()
        self.stats.start_time = time.time()
        
        # Dữ liệu đã crawl
        self.crawled_posts: List[Dict] = []
        
        logger.info(f"Initialized crawler with proxy rotation every {proxy_rotation_interval} requests")
        logger.info(f"Output directory: {self.output_dir}")
        
        if use_proxy:
            proxy_stats = self.client.get_proxy_statistics()
            logger.info(f"Proxy manager status: {proxy_stats.get('total_proxies', 0)} proxies available")
    
    def _extract_post(self, post_data: Dict, subreddit: str) -> Dict:
        """Trích xuất thông tin bài viết từ JSON response"""
        created_utc = post_data.get('created_utc', 0)
        
        return {
            'id': post_data.get('id'),
            'subreddit': subreddit,
            'title': post_data.get('title', ''),
            'content': post_data.get('selftext', '')[:2000],  # Giới hạn 2000 ký tự
            'author': post_data.get('author', '[deleted]'),
            'score': post_data.get('score', 0),
            'upvote_ratio': post_data.get('upvote_ratio', 0),
            'num_comments': post_data.get('num_comments', 0),
            'created_utc': created_utc,
            'created_date': datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S') if created_utc else None,
            'permalink': post_data.get('permalink'),
            'url': f"https://reddit.com{post_data.get('permalink', '')}",
            'flair': post_data.get('link_flair_text'),
            'over_18': post_data.get('over_18', False),
            'is_self': post_data.get('is_self', True),
            'crawled_at': datetime.now().isoformat()
        }
    
    def _save_checkpoint(self, subreddit: str, posts: List[Dict]) -> None:
        """Lưu checkpoint trong quá trình crawl"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"checkpoint_{subreddit}.json"
        
        checkpoint_data = {
            'subreddit': subreddit,
            'total_posts': len(posts),
            'last_update': datetime.now().isoformat(),
            'recent_posts': posts[-self.checkpoint_interval:] if posts else [],
            'stats': {
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests
            }
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  💾 Checkpoint saved: {len(posts)} posts so far")
    
    def _save_results(self, subreddit: str, posts: List[Dict]) -> None:
        """Lưu kết quả cuối cùng"""
        if not posts:
            logger.warning(f"No posts to save for r/{subreddit}")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Lưu JSON
        json_file = self.output_dir / f"{subreddit}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  💾 Saved JSON: {json_file} ({len(posts)} posts)")
        
        # Lưu CSV (nếu có pandas)
        try:
            import pandas as pd
            csv_file = self.output_dir / f"{subreddit}_{timestamp}.csv"
            df = pd.DataFrame(posts)
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            logger.info(f"  💾 Saved CSV: {csv_file}")
        except ImportError:
            logger.debug("Pandas not available, skipping CSV export")
    
    def crawl_subreddit(self, subreddit: str, limit: int = None) -> List[Dict]:
        """
        Crawl một subreddit với proxy rotation
        
        Args:
            subreddit: Tên subreddit (không bao gồm r/)
            limit: Số bài viết tối đa
        
        Returns:
            List[Dict]: Danh sách bài viết
        """
        if limit is None:
            limit = self.max_posts_per_subreddit
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Crawling r/{subreddit}")
        logger.info(f"   Max posts: {limit}")
        logger.info(f"   Proxy rotation: every {self.rotation_interval} requests")
        logger.info(f"{'='*60}")
        
        posts = []
        after = None
        page = 0
        failed_pages = 0
        max_failed_pages = 5
        
        while len(posts) < limit and failed_pages < max_failed_pages:
            page += 1
            
            # Build URL
            url = f"https://www.reddit.com/r/{subreddit}/new/.json"
            params = f"?limit={min(100, limit - len(posts))}"
            if after:
                params += f"&after={after}"
            
            full_url = url + params
            
            # Log request info
            logger.info(f"  📡 Page {page}: fetching {min(100, limit - len(posts))} posts...")
            self.stats.total_requests += 1
            
            # Make request
            data = self.client.get_json(full_url)
            
            if not data:
                failed_pages += 1
                logger.warning(f"  ❌ Failed to fetch page {page} (attempt {failed_pages}/{max_failed_pages})")
                time.sleep(self.delay * 2)
                continue
            
            # Reset failed counter on success
            failed_pages = 0
            self.stats.successful_requests += 1
            
            # Parse response
            try:
                children = data.get('data', {}).get('children', [])
                
                if not children:
                    logger.info(f"  📭 No more posts available")
                    break
                
                # Extract post data
                page_posts = []
                for child in children:
                    post_data = child.get('data', {})
                    post = self._extract_post(post_data, subreddit)
                    page_posts.append(post)
                    posts.append(post)
                
                # Update stats
                self.stats.total_posts += len(page_posts)
                
                # Get next page token
                after = data.get('data', {}).get('after')
                
                logger.info(f"  ✅ Page {page}: got {len(page_posts)} posts (total: {len(posts)}/{limit})")
                
                # Save checkpoint
                if len(posts) % self.checkpoint_interval == 0:
                    self._save_checkpoint(subreddit, posts)
                
                # If no next page, break
                if not after:
                    logger.info(f"  📭 No more pages available")
                    break
                
            except json.JSONDecodeError as e:
                logger.error(f"  ❌ JSON decode error: {e}")
                failed_pages += 1
                
            except Exception as e:
                logger.error(f"  ❌ Parse error: {e}")
                failed_pages += 1
            
            # Delay between requests
            if len(posts) < limit:
                time.sleep(self.delay)
        
        if failed_pages >= max_failed_pages:
            logger.warning(f"  ⚠️ Stopped after {max_failed_pages} consecutive failures")
        
        # Update subreddit stats
        self.stats.subreddit_stats[subreddit] = len(posts)
        
        # Save final results
        self._save_results(subreddit, posts)
        
        logger.info(f"  📊 Completed r/{subreddit}: {len(posts)} posts collected")
        
        return posts
    
    def crawl_multiple_subreddits(
        self,
        subreddits: List[str] = None,
        limit_per_subreddit: int = None
    ) -> Dict[str, List[Dict]]:
        """
        Crawl nhiều subreddit với proxy rotation
        
        Args:
            subreddits: Danh sách subreddit (None = dùng mặc định)
            limit_per_subreddit: Số bài mỗi subreddit
        
        Returns:
            Dict: {subreddit: posts}
        """
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS
        
        if limit_per_subreddit is None:
            limit_per_subreddit = self.max_posts_per_subreddit
        
        results = {}
        
        for i, subreddit in enumerate(subreddits, 1):
            logger.info(f"\n{'#'*60}")
            logger.info(f"📌 [{i}/{len(subreddits)}] Processing r/{subreddit}")
            logger.info(f"{'#'*60}")
            
            # Reset proxy counter for each subreddit (optional)
            self.client.reset_proxy_counter()
            
            # Crawl subreddit
            posts = self.crawl_subreddit(subreddit, limit=limit_per_subreddit)
            results[subreddit] = posts
            
            # Nghỉ giữa các subreddit
            if i < len(subreddits):
                wait_time = 10
                logger.info(f"⏸️  Waiting {wait_time} seconds before next subreddit...")
                time.sleep(wait_time)
        
        return results
    
    def save_summary_report(self) -> None:
        """Lưu báo cáo tổng kết"""
        self.stats.end_time = time.time()
        
        report_file = self.output_dir / f"crawl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'summary': self.stats.to_dict(),
            'proxy_statistics': self.client.get_proxy_statistics() if self.client.use_proxy else None,
            'config': {
                'rotation_interval': self.rotation_interval,
                'delay': self.delay,
                'max_posts_per_subreddit': self.max_posts_per_subreddit,
                'checkpoint_interval': self.checkpoint_interval
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Summary report saved: {report_file}")
    
    def print_statistics(self) -> None:
        """In thống kê cuối cùng ra console"""
        self.stats.end_time = time.time()
        
        print("\n" + "="*70)
        print("📊 CRAWL STATISTICS")
        print("="*70)
        print(f"⏱️  Time elapsed: {self.stats.elapsed_seconds:.1f} seconds")
        print(f"📡 Total requests: {self.stats.total_requests}")
        print(f"✅ Successful: {self.stats.successful_requests}")
        print(f"❌ Failed: {self.stats.failed_requests}")
        print(f"📈 Success rate: {self.stats.success_rate:.1%}")
        print(f"📝 Total posts: {self.stats.total_posts}")
        
        print("\n📋 Posts per subreddit:")
        for subreddit, count in sorted(self.stats.subreddit_stats.items(), key=lambda x: -x[1]):
            print(f"   - r/{subreddit}: {count} posts")
        
        if self.stats.proxies_used:
            print("\n🔄 Proxies used:")
            for proxy, count in sorted(self.stats.proxies_used.items(), key=lambda x: -x[1])[:10]:
                proxy_short = proxy[:50] + "..." if len(proxy) > 50 else proxy
                print(f"   - {proxy_short}: {count} requests")
        
        # Proxy statistics from client
        if self.client.use_proxy:
            proxy_stats = self.client.get_proxy_statistics()
            print("\n📡 Proxy Manager Status:")
            print(f"   - Total proxies: {proxy_stats.get('total_proxies', 0)}")
            print(f"   - Alive proxies: {proxy_stats.get('alive_proxies', 0)}")
            print(f"   - Rotation strategy: {proxy_stats.get('rotation_strategy', 'N/A')}")
            print(f"   - Requests since last rotation: {proxy_stats.get('request_counter', 0)}/{proxy_stats.get('rotation_interval', 'N/A')}")
        
        print("\n" + "="*70)
    
    def resume_from_checkpoint(self, subreddit: str) -> Optional[List[Dict]]:
        """
        Tiếp tục crawl từ checkpoint (nếu có)
        
        Args:
            subreddit: Tên subreddit cần resume
        
        Returns:
            Danh sách posts đã crawl trước đó hoặc None
        """
        checkpoint_file = self.output_dir / "checkpoints" / f"checkpoint_{subreddit}.json"
        
        if not checkpoint_file.exists():
            logger.info(f"No checkpoint found for r/{subreddit}")
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            posts = checkpoint.get('recent_posts', [])
            logger.info(f"Resumed r/{subreddit} from checkpoint: {len(posts)} posts already crawled")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function để chạy crawler"""
    
    parser = argparse.ArgumentParser(
        description="Reddit Crawler with Proxy Rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl all default subreddits
  python crawl_with_proxy_rotation.py
  
  # Crawl specific subreddits
  python crawl_with_proxy_rotation.py --subreddits depression anxiety --limit 200
  
  # Quick test with 1 subreddit, 20 posts
  python crawl_with_proxy_rotation.py --test
  
  # Crawl without proxy
  python crawl_with_proxy_rotation.py --no-proxy
  
  # Custom rotation interval
  python crawl_with_proxy_rotation.py --rotation 50 --delay 3.0
        """
    )
    
    parser.add_argument(
        "--subreddits", "-s",
        nargs="+",
        help="List of subreddits to crawl (default: depression, anxiety, SuicideWatch, lonely, offmychest, mentalhealth)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=300,
        help="Maximum posts per subreddit (default: 300)"
    )
    
    parser.add_argument(
        "--rotation", "-r",
        type=int,
        default=100,
        help="Proxy rotation interval (requests before rotating, default: 100)"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)"
    )
    
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable proxy (crawl directly)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="data/raw/reddit",
        help="Output directory (default: data/raw/reddit)"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run quick test mode (1 subreddit, 20 posts)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    
    parser.add_argument(
        "--refresh-proxies",
        action="store_true",
        help="Force refresh proxy list before starting"
    )
    
    args = parser.parse_args()
    
    # In banner
    print("="*70)
    print("🐍 REDDIT CRAWLER WITH PROXY ROTATION")
    print("="*70)
    print(f"⚙️  Configuration:")
    print(f"   - Proxy rotation: every {args.rotation} requests")
    print(f"   - Delay between requests: {args.delay} seconds")
    print(f"   - Max posts per subreddit: {args.limit}")
    print(f"   - Use proxy: {not args.no_proxy}")
    print(f"   - Output directory: {args.output}")
    print("="*70)
    
    # Khởi tạo crawler
    crawler = RedditCrawlerWithRotation(
        proxy_rotation_interval=args.rotation,
        delay_between_requests=args.delay,
        output_dir=args.output,
        checkpoint_interval=50,
        max_posts_per_subreddit=args.limit,
        use_proxy=not args.no_proxy,
        max_retries=3
    )
    
    # Refresh proxies if requested
    if args.refresh_proxies and not args.no_proxy:
        logger.info("Refreshing proxy list...")
        crawler.client.refresh_proxies()
    
    # Quick test mode
    if args.test:
        logger.info("\n🧪 Running in TEST mode (1 subreddit, 20 posts)")
        test_subreddits = ["depression"]
        results = crawler.crawl_multiple_subreddits(
            subreddits=test_subreddits,
            limit_per_subreddit=20
        )
        crawler.print_statistics()
        return
    
    # Xác định subreddits cần crawl
    if args.subreddits:
        subreddits = args.subreddits
        logger.info(f"Target subreddits: {', '.join(subreddits)}")
    else:
        subreddits = crawler.DEFAULT_SUBREDDITS
        logger.info(f"Target subreddits (default): {', '.join(subreddits)}")
    
    # Confirm before crawling
    print("\n⚠️  Note: Crawling may take 10-60 minutes depending on network and settings")
    confirm = input("\n👉 Start crawling? (y/n): ").strip().lower()
    
    if confirm != 'y' and confirm != 'yes':
        print("Cancelled.")
        return
    
    # Chạy crawl
    try:
        results = crawler.crawl_multiple_subreddits(
            subreddits=subreddits,
            limit_per_subreddit=args.limit
        )
        
        # Lưu báo cáo và in thống kê
        crawler.save_summary_report()
        crawler.print_statistics()
        
        # Tổng kết
        total_posts = sum(len(posts) for posts in results.values())
        print(f"\n🎉 CRAWL COMPLETED!")
        print(f"📊 Total posts collected: {total_posts}")
        print(f"📁 Data saved to: {crawler.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Crawl interrupted by user")
        crawler.print_statistics()
        crawler.save_summary_report()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Crawl failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()