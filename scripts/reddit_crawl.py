#!/usr/bin/env python3
"""
Script to crawl Reddit data for depression detection system
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import RedditCrawler
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Crawl Reddit for depression-related content")
    
    parser.add_argument(
        "--subreddits", "-s",
        nargs="+",
        help="List of subreddits (default: all depression-related)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=300,
        help="Posts per subreddit (default: 300)"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=2.0,
        help="Delay between requests (seconds)"
    )
    
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use proxy rotation"
    )
    
    parser.add_argument(
        "--comments",
        action="store_true",
        help="Include comments (slower, more data)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="data/raw/reddit",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(log_level="INFO")
    
    # Initialize crawler
    crawler = RedditCrawler(
        delay=args.delay,
        use_proxy=args.proxy,
        output_dir=args.output
    )
    
    # Determine subreddits
    if args.subreddits:
        subreddits = args.subreddits
    else:
        subreddits = crawler.TARGET_SUBREDDITS
    
    logger.info(f"Target subreddits: {subreddits}")
    logger.info(f"Limit per subreddit: {args.limit}")
    logger.info(f"Include comments: {args.comments}")
    
    # Crawl
    results = {}
    for subreddit in subreddits:
        posts = crawler.crawl_subreddit(
            subreddit=subreddit,
            limit=args.limit,
            include_comments=args.comments
        )
        results[subreddit] = posts
    
    # Summary
    print("\n" + "="*50)
    print("CRAWL SUMMARY")
    print("="*50)
    for subreddit, posts in results.items():
        print(f"r/{subreddit}: {len(posts)} posts")
    
    total_posts = sum(len(p) for p in results.values())
    print(f"\nTotal: {total_posts} posts collected")
    print(f"Data saved to: {args.output}")


if __name__ == "__main__":
    main()