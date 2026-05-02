"""
Parse Reddit JSON responses into clean dictionary format
"""

from typing import List, Dict, Optional
from datetime import datetime


class RedditParser:
    """Parse Reddit API JSON responses"""
    
    @staticmethod
    def parse_listing(data: Dict) -> List[Dict]:
        """
        Parse a listing response (multiple posts)
        
        Returns list of posts with standardized fields
        """
        posts = []
        
        try:
            children = data.get('data', {}).get('children', [])
            
            for child in children:
                post_data = child.get('data', {})
                parsed = RedditParser._parse_post(post_data)
                if parsed:
                    posts.append(parsed)
                    
        except Exception as e:
            print(f"Error parsing listing: {e}")
        
        return posts
    
    @staticmethod
    def parse_single_post(data: Dict) -> Optional[Dict]:
        """Parse a single post response"""
        try:
            # Response structure for single post: [{listing}, {comments}]
            if isinstance(data, list) and len(data) > 0:
                post_listing = data[0]
                post_data = post_listing.get('data', {}).get('children', [{}])[0].get('data', {})
                return RedditParser._parse_post(post_data)
        except Exception as e:
            print(f"Error parsing post: {e}")
        
        return None
    
    @staticmethod
    def parse_comments(data: Dict, limit: int = 500) -> List[Dict]:
        """Parse comments from response"""
        comments = []
        
        try:
            # Comments are in second element of response
            if isinstance(data, list) and len(data) > 1:
                comments_listing = data[1]
                children = comments_listing.get('data', {}).get('children', [])
                
                for child in children:
                    if len(comments) >= limit:
                        break
                    
                    # Skip MoreComments placeholder
                    if child.get('kind') == 'more':
                        continue
                    
                    comment_data = child.get('data', {})
                    
                    # Skip deleted comments
                    if comment_data.get('body') in ['[deleted]', '[removed]']:
                        continue
                    
                    parsed = RedditParser._parse_comment(comment_data)
                    if parsed:
                        comments.append(parsed)
                        
        except Exception as e:
            print(f"Error parsing comments: {e}")
        
        return comments
    
    @staticmethod
    def _parse_post(data: Dict) -> Dict:
        """Internal: Convert raw post data to standardized format"""
        created_utc = data.get('created_utc', 0)
        
        return {
            'id': data.get('id'),
            'subreddit': data.get('subreddit'),
            'title': data.get('title'),
            'content': data.get('selftext', '')[:2000],  # Limit length
            'author': data.get('author'),
            'score': data.get('score'),
            'upvote_ratio': data.get('upvote_ratio'),
            'num_comments': data.get('num_comments'),
            'created_utc': created_utc,
            'created_date': datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'permalink': data.get('permalink'),
            'url': f"https://reddit.com{data.get('permalink')}",
            'flair': data.get('link_flair_text'),
            'is_self': data.get('is_self'),
            'over_18': data.get('over_18')
        }
    
    @staticmethod
    def _parse_comment(data: Dict) -> Dict:
        """Internal: Convert raw comment data to standardized format"""
        created_utc = data.get('created_utc', 0)
        
        return {
            'id': data.get('id'),
            'post_id': data.get('link_id', '').replace('t3_', ''),
            'author': data.get('author'),
            'body': data.get('body', '')[:1000],
            'score': data.get('score'),
            'created_utc': created_utc,
            'created_date': datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'parent_id': data.get('parent_id'),
            'depth': data.get('depth', 0),
            'is_submitter': data.get('is_submitter', False)
        }