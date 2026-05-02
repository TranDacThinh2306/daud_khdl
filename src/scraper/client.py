"""
Low-level Reddit API client (no authentication required)
"""

import requests
import time
import random
from typing import Optional, Dict, List
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger("depression_alert.scraper.client")

# Import RateLimiter và ProxyManager
from src.scraper.rate_limiter import RateLimiter
from src.scraper.proxy import ProxyManager


class RedditClient:
    """Handles HTTP requests to Reddit's .json endpoint with proxy support"""
    
    def __init__(
        self,
        delay: float = 2.0,
        use_proxy: bool = False,
        max_retries: int = 3,
        proxy_config: Optional[Dict] = None,
        proxy_rotation_interval: Optional[int] = None
    ):
        """
        Khởi tạo Reddit Client
        
        Args:
            delay: Delay giữa các request (giây)
            use_proxy: Có sử dụng proxy không
            max_retries: Số lần thử lại tối đa
            proxy_config: Cấu hình cho ProxyManager (dict)
            proxy_rotation_interval: Số request trước khi xoay proxy (optional)
        """
        self.delay = delay
        self.use_proxy = use_proxy
        self.max_retries = max_retries
        self.proxy_rotation_interval = proxy_rotation_interval
        self.request_counter = 0
        self.current_proxy = None
        
        # Khởi tạo rate limiter
        self.rate_limiter = RateLimiter(min_interval=delay)
        
        # Khởi tạo proxy manager nếu cần
        if use_proxy:
            default_config = {
                'auto_fetch_free': True,
                'check_on_start': False,      # Không check khi khởi tạo để tránh nghẽn
                'lazy_check': True,           # Chỉ check khi cần
                'max_failures': 3,
                'rotation_strategy': 'round_robin',
                'check_timeout': 10,
                'max_check_workers': 10,
                'min_proxies': 3
            }
            
            if proxy_config:
                default_config.update(proxy_config)
            
            self.proxy_manager = ProxyManager(**default_config)
            logger.info(f"ProxyManager initialized with strategy: {default_config['rotation_strategy']}")
        else:
            self.proxy_manager = None
        
        # User agents để xoay vòng
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.1 Safari/605.1.15"
        ]
        
        logger.info(f"RedditClient initialized (proxy={use_proxy}, delay={delay}s, max_retries={max_retries})")
    
    def _get_headers(self) -> Dict[str, str]:
        """Tạo headers với User-Agent ngẫu nhiên"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.reddit.com/",
            "Connection": "keep-alive"
        }
    
    def _should_rotate_proxy(self) -> bool:
        """Kiểm tra có cần xoay proxy không"""
        if not self.use_proxy or not self.proxy_rotation_interval:
            return False
        return self.request_counter >= self.proxy_rotation_interval
    
    def _rotate_proxy(self) -> Optional[Dict[str, str]]:
        """Xoay sang proxy mới"""
        if not self.use_proxy or not self.proxy_manager:
            return None
        
        logger.info(f"🔄 Rotating proxy after {self.request_counter} requests")
        self.request_counter = 0
        
        new_proxy = self.proxy_manager.get_next_proxy()
        
        if new_proxy:
            self.current_proxy = new_proxy
            proxy_addr = new_proxy.get('http', 'unknown')
            logger.debug(f"Switched to proxy: {proxy_addr[:50]}")
        else:
            logger.warning("No proxy available, continuing without proxy")
            self.current_proxy = None
        
        return self.current_proxy
    
    def _get_proxy(self) -> Optional[Dict[str, str]]:
        """Lấy proxy hiện tại hoặc xoay nếu cần"""
        if not self.use_proxy or not self.proxy_manager:
            return None
        
        # Kiểm tra rotation
        if self._should_rotate_proxy():
            return self._rotate_proxy()
        
        # Lấy proxy mới nếu chưa có
        if self.current_proxy is None:
            self.current_proxy = self.proxy_manager.get_next_proxy()
            if self.current_proxy:
                logger.debug(f"Initial proxy: {self.current_proxy.get('http', 'unknown')[:50]}")
        
        return self.current_proxy
    
    def _mark_proxy_result(self, proxy: Optional[Dict[str, str]], success: bool) -> None:
        """Đánh dấu kết quả của proxy"""
        if not proxy or not self.use_proxy or not self.proxy_manager:
            return
        
        if success:
            self.proxy_manager.mark_proxy_success(proxy)
        else:
            self.proxy_manager.mark_proxy_failed(proxy)
            # Nếu fail, xóa proxy hiện tại để lấy mới lần sau
            self.current_proxy = None
    
    def get_json(self, url: str, forced_proxy: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """
        Fetch JSON từ Reddit .json endpoint với hỗ trợ proxy
        
        Args:
            url: URL cần request
            forced_proxy: Proxy cố định (bỏ qua rotation)
        
        Returns:
            JSON data hoặc None nếu thất bại
        """
        for attempt in range(self.max_retries):
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Lấy proxy
            proxies = forced_proxy if forced_proxy else self._get_proxy()
            
            # Log thông tin request
            proxy_info = "no proxy"
            if proxies:
                proxy_info = proxies.get('http', 'unknown')[:50]
            logger.debug(f"Request attempt {attempt + 1}/{self.max_retries} via {proxy_info}")
            
            try:
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    proxies=proxies,
                    timeout=30,
                    allow_redirects=True
                )
                
                # Xử lý response
                if response.status_code == 200:
                    logger.debug(f"Success: {url[:80]}...")
                    self._mark_proxy_result(proxies, success=True)
                    self.request_counter += 1
                    return response.json()
                
                elif response.status_code == 429:
                    # Rate limit - đánh dấu proxy fail và chờ
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limited (429), waiting {wait_time}s")
                    self._mark_proxy_result(proxies, success=False)
                    time.sleep(wait_time)
                
                elif response.status_code == 403:
                    # Bị chặn - đánh dấu proxy fail và thử proxy khác
                    logger.warning(f"Blocked (403) with proxy {proxy_info}")
                    self._mark_proxy_result(proxies, success=False)
                    
                    # Force rotate proxy ngay lập tức
                    if self.use_proxy and self.proxy_manager:
                        self.current_proxy = None
                        self.request_counter = self.proxy_rotation_interval  # Force rotate
                    
                elif response.status_code == 404:
                    logger.error(f"Not found (404): {url}")
                    return None
                    
                else:
                    logger.error(f"HTTP {response.status_code}: {url}")
                    self._mark_proxy_result(proxies, success=False)
                    
            except requests.exceptions.Timeout:
                logger.error(f"Timeout (attempt {attempt + 1}): {url}")
                self._mark_proxy_result(proxies, success=False)
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error (attempt {attempt + 1}): {e}")
                self._mark_proxy_result(proxies, success=False)
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                self._mark_proxy_result(proxies, success=False)
            
            # Chờ trước khi thử lại
            if attempt < self.max_retries - 1:
                wait_time = self.delay * (attempt + 1)
                logger.debug(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        logger.error(f"All {self.max_retries} attempts failed for {url[:80]}")
        return None
    
    def get_json_with_retry(
        self, 
        url: str, 
        max_retries: Optional[int] = None,
        force_new_proxy: bool = False
    ) -> Optional[Dict]:
        """
        Fetch JSON với số lần thử lại tùy chỉnh
        
        Args:
            url: URL cần request
            max_retries: Số lần thử lại tối đa (override default)
            force_new_proxy: Ép xoay proxy mới
        """
        if force_new_proxy and self.use_proxy:
            self.current_proxy = None
            self.request_counter = self.proxy_rotation_interval if self.proxy_rotation_interval else 0
        
        original_retries = self.max_retries
        if max_retries:
            self.max_retries = max_retries
        
        result = self.get_json(url)
        
        # Restore
        self.max_retries = original_retries
        
        return result
    
    def get_proxy_statistics(self) -> Dict:
        """Lấy thống kê về proxy đang sử dụng"""
        if not self.use_proxy or not self.proxy_manager:
            return {"proxy_enabled": False}
        
        stats = self.proxy_manager.get_statistics()
        stats.update({
            "request_counter": self.request_counter,
            "rotation_interval": self.proxy_rotation_interval,
            "current_proxy": self.current_proxy.get('http', 'none')[:50] if self.current_proxy else None
        })
        
        return stats
    
    def reset_proxy_counter(self) -> None:
        """Reset counter cho rotation"""
        self.request_counter = 0
        logger.info("Proxy request counter reset")
    
    def refresh_proxies(self) -> None:
        """Làm mới danh sách proxy"""
        if self.use_proxy and self.proxy_manager:
            logger.info("Manually refreshing proxy list...")
            self.proxy_manager.refresh_proxies(force=True)
            self.current_proxy = None
            self.request_counter = 0


class RateLimiter:
    """Đơn giản hóa rate limiter (nếu file rate_limiter.py chưa có)"""
    
    def __init__(self, min_interval: float = 2.0):
        self.min_interval = min_interval
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Chờ nếu cần để đảm bảo rate limit"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


# ============================================
# VÍ DỤ SỬ DỤNG
# ============================================

def demo():
    """Demo cách sử dụng RedditClient với proxy"""
    
    # Cách 1: Không dùng proxy
    print("=" * 50)
    print("DEMO 1: No proxy")
    print("=" * 50)
    
    client1 = RedditClient(
        delay=1.0,
        use_proxy=False,
        max_retries=2
    )
    
    url = "https://www.reddit.com/r/all/new/.json?limit=1"
    result = client1.get_json(url)
    
    if result:
        print("✅ Successfully fetched data without proxy")
    else:
        print("❌ Failed to fetch data")
    
    # Cách 2: Dùng proxy với rotation mỗi 5 request
    print("\n" + "=" * 50)
    print("DEMO 2: With proxy rotation (every 5 requests)")
    print("=" * 50)
    
    client2 = RedditClient(
        delay=2.0,
        use_proxy=True,
        max_retries=3,
        proxy_rotation_interval=5,  # Xoay sau mỗi 5 request
        proxy_config={
            'auto_fetch_free': True,
            'check_on_start': False,
            'lazy_check': True,
            'rotation_strategy': 'random'
        }
    )
    
    # Thực hiện vài request
    for i in range(3):
        print(f"\n--- Request {i+1} ---")
        result = client2.get_json(url)
        if result:
            print(f"✅ Request {i+1} successful")
            stats = client2.get_proxy_statistics()
            print(f"   Request counter: {stats['request_counter']}/{stats['rotation_interval']}")
        else:
            print(f"❌ Request {i+1} failed")
        time.sleep(1)
    
    # In thống kê
    print("\n" + "=" * 50)
    print("PROXY STATISTICS")
    print("=" * 50)
    stats = client2.get_proxy_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()