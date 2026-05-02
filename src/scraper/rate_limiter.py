# src/scraper/rate_limiter.py

import time
from typing import Optional


class RateLimiter:
    """Rate limiter để kiểm soát frequency của requests"""
    
    def __init__(self, min_interval: float = 2.0):
        """
        Args:
            min_interval: Thời gian tối thiểu giữa các request (giây)
        """
        self.min_interval = min_interval
        self.last_request_time: Optional[float] = None
    
    def wait_if_needed(self) -> None:
        """Chờ nếu cần để đảm bảo rate limit"""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def reset(self) -> None:
        """Reset last request time"""
        self.last_request_time = None