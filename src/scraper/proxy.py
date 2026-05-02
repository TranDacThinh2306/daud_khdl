"""
Proxy Manager for Reddit Scraper - Cải tiến với concurrent checking
"""

import requests
import time
import random
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import logger
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("depression_xai.scraper.proxy_manager")


@dataclass
class ProxyInfo:
    """Thông tin chi tiết về một proxy"""
    address: str
    protocol: str
    ip: str
    port: int
    country: Optional[str] = None
    anonymity: Optional[str] = None
    speed: Optional[float] = None
    last_used: Optional[datetime] = None
    success_count: int = 0
    fail_count: int = 0
    last_check: Optional[datetime] = None
    is_alive: bool = True
    is_checking: bool = False  # Đang được kiểm tra
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def is_reliable(self) -> bool:
        return self.success_rate > 0.7 and self.success_count > 5


class ProxyChecker:
    """Kiểm tra proxy với concurrent execution"""
    
    TEST_URLS = [
        'http://httpbin.org/ip',
        'http://httpbin.org/get',
    ]
    
    def __init__(self, timeout: int = 10, max_workers: int = 10):
        """
        Args:
            timeout: Timeout cho mỗi request (giây)
            max_workers: Số luồng kiểm tra đồng thời
        """
        self.timeout = timeout
        self.max_workers = max_workers
    
    def check_single_proxy(self, proxy_address: str) -> Tuple[str, bool, Optional[float]]:
        """
        Kiểm tra một proxy (dùng cho concurrent)
        
        Returns:
            (proxy_address, is_alive, response_time)
        """
        proxies = {'http': proxy_address, 'https': proxy_address}
        
        for test_url in self.TEST_URLS:
            try:
                start_time = time.time()
                response = requests.get(
                    test_url,
                    proxies=proxies,
                    timeout=self.timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    return proxy_address, True, response_time
                    
            except Exception:
                continue
        
        return proxy_address, False, None
    
    def check_proxies_concurrent(self, proxy_addresses: List[str]) -> Dict[str, Tuple[bool, Optional[float]]]:
        """
        Kiểm tra nhiều proxy cùng lúc (concurrent)
        
        Returns:
            Dict {proxy_address: (is_alive, response_time)}
        """
        if not proxy_addresses:
            return {}
        
        logger.info(f"Checking {len(proxy_addresses)} proxies concurrently with {self.max_workers} workers...")
        
        results = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_proxy = {
                executor.submit(self.check_single_proxy, proxy): proxy 
                for proxy in proxy_addresses
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_proxy):
                proxy = future_to_proxy[future]
                try:
                    proxy_addr, is_alive, speed = future.result(timeout=self.timeout + 5)
                    results[proxy_addr] = (is_alive, speed)
                except Exception as e:
                    logger.debug(f"Proxy {proxy} check failed: {e}")
                    results[proxy] = (False, None)
                
                completed += 1
                if completed % 10 == 0:
                    logger.debug(f"Proxy check progress: {completed}/{len(proxy_addresses)}")
        
        alive_count = sum(1 for v in results.values() if v[0])
        logger.info(f"Concurrent check complete: {alive_count}/{len(proxy_addresses)} proxies alive")
        
        return results


class ProxyManager:
    """Quản lý proxy với concurrent checking và async refresh"""
    
    def __init__(
        self,
        initial_proxies: Optional[List[str]] = None,
        auto_fetch_free: bool = True,
        check_on_start: bool = False,  # Mặc định là False để tránh nghẽn
        max_failures: int = 3,
        rotation_strategy: str = "round_robin",
        refresh_interval_minutes: int = 30,
        min_proxies: int = 3,
        check_timeout: int = 10,
        max_check_workers: int = 20,
        lazy_check: bool = True  # Chỉ check proxy khi cần dùng
    ):
        """
        Args:
            lazy_check: Chỉ kiểm tra proxy khi thực sự cần dùng (recommended)
            check_timeout: Timeout cho mỗi proxy check (giây)
            max_check_workers: Số luồng kiểm tra đồng thời
        """
        self.proxies: List[ProxyInfo] = []
        self.proxy_index = 0
        self.max_failures = max_failures
        self.rotation_strategy = rotation_strategy
        self.refresh_interval = timedelta(minutes=refresh_interval_minutes)
        self.min_proxies = min_proxies
        self.lazy_check = lazy_check
        self.last_refresh = datetime.now()
        
        self.checker = ProxyChecker(timeout=check_timeout, max_workers=max_check_workers)
        self._lock = threading.Lock()
        self._is_checking = False
        
        logger.info(f"Initializing ProxyManager (lazy_check={lazy_check}, strategy={rotation_strategy})")
        
        # Khởi tạo danh sách proxy
        if initial_proxies:
            self.add_proxies(initial_proxies)
        
        if auto_fetch_free:
            self.fetch_and_add_free_proxies()
        
        # Chỉ check nếu được yêu cầu và KHÔNG dùng lazy_check
        if check_on_start and not lazy_check and self.proxies:
            logger.info("Checking all proxies on start (this may take a while)...")
            self.check_all_proxies_concurrent()
        
        alive_count = len(self.get_alive_proxies())
        logger.info(f"ProxyManager initialized with {alive_count}/{len(self.proxies)} alive proxies")
        
        # Nếu không có proxy alive và có proxy, tự động check
        if alive_count == 0 and self.proxies:
            logger.warning("No alive proxies, triggering background check...")
            self.check_all_proxies_background()
    
    def _parse_proxy_address(self, address: str) -> ProxyInfo:
        """Parse địa chỉ proxy"""
        protocol = 'http'
        if '://' in address:
            protocol = address.split('://')[0]
            address = address.split('://')[1]
        
        if ':' in address:
            ip, port_str = address.rsplit(':', 1)
            port = int(port_str)
        else:
            ip = address
            port = 8080
        
        return ProxyInfo(
            address=f"{protocol}://{ip}:{port}",
            protocol=protocol,
            ip=ip,
            port=port
        )
    
    def add_proxy(self, proxy_address: str) -> None:
        """Thêm một proxy vào danh sách"""
        with self._lock:
            existing = [p for p in self.proxies if p.address == proxy_address]
            if existing:
                return
            
            proxy_info = self._parse_proxy_address(proxy_address)
            self.proxies.append(proxy_info)
            logger.debug(f"Added proxy: {proxy_address}")
    
    def add_proxies(self, proxy_addresses: List[str]) -> None:
        """Thêm nhiều proxy cùng lúc"""
        for proxy in proxy_addresses:
            self.add_proxy(proxy)
    
    def remove_proxy(self, proxy_address: str) -> None:
        """Xóa proxy"""
        with self._lock:
            before_count = len(self.proxies)
            self.proxies = [p for p in self.proxies if p.address != proxy_address]
            if before_count != len(self.proxies):
                logger.debug(f"Removed proxy: {proxy_address}")
    
    def fetch_and_add_free_proxies(self) -> int:
        """Lấy proxy free và thêm vào danh sách"""
        logger.info("Fetching free proxies from internet...")
        
        # Giới hạn số lượng proxy free để tránh quá tải
        free_proxies = self._fetch_limited_proxies(max_proxies=50)
        
        before_count = len(self.proxies)
        self.add_proxies(free_proxies)
        after_count = len(self.proxies)
        
        added = after_count - before_count
        logger.info(f"Added {added} new free proxies")
        
        return added
    
    def _fetch_limited_proxies(self, max_proxies: int = 50) -> List[str]:
        """Lấy proxy free có giới hạn số lượng"""
        all_proxies = []
        
        sources = [
            self._fetch_from_proxyroll,
            self._fetch_from_proxyscrape,
        ]
        
        for source in sources:
            if len(all_proxies) >= max_proxies:
                break
            try:
                proxies = source()
                all_proxies.extend(proxies[:max_proxies - len(all_proxies)])
            except Exception as e:
                logger.warning(f"Source failed: {e}")
        
        return list(set(all_proxies))[:max_proxies]
    
    def _fetch_from_proxyroll(self) -> List[str]:
        """Lấy proxy từ ProxyRoll"""
        try:
            url = 'https://api.proxyroll.com/v2/?get=true&https=true&anonymity=true'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                proxies = response.text.strip().split('\n')
                return [f"http://{p}" for p in proxies if p]
        except Exception as e:
            logger.warning(f"ProxyRoll failed: {e}")
        return []
    
    def _fetch_from_proxyscrape(self) -> List[str]:
        """Lấy proxy từ ProxyScrape"""
        try:
            url = 'https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                proxies = response.text.strip().split('\r\n')
                return [f"http://{p}" for p in proxies if p]
        except Exception as e:
            logger.warning(f"ProxyScrape failed: {e}")
        return []
    
    def check_all_proxies_concurrent(self) -> Dict[str, bool]:
        """Kiểm tra tất cả proxy concurrent - KHÔNG bị nghẽn"""
        if self._is_checking:
            logger.warning("Proxy check already in progress, skipping")
            return {}
        
        with self._lock:
            if not self.proxies:
                return {}
            
            self._is_checking = True
            proxy_addresses = [p.address for p in self.proxies]
        
        try:
            logger.info(f"Checking {len(proxy_addresses)} proxies concurrently...")
            start_time = time.time()
            
            # Concurrent check
            results = self.checker.check_proxies_concurrent(proxy_addresses)
            
            # Cập nhật kết quả
            with self._lock:
                for proxy in self.proxies:
                    if proxy.address in results:
                        is_alive, speed = results[proxy.address]
                        proxy.is_alive = is_alive
                        proxy.speed = speed
                        proxy.last_check = datetime.now()
                        if is_alive:
                            proxy.fail_count = 0
            
            elapsed = time.time() - start_time
            alive_count = sum(1 for v in results.values() if v[0])
            logger.info(f"Concurrent check completed in {elapsed:.1f}s: {alive_count}/{len(proxy_addresses)} alive")
            
            return {addr: is_alive for addr, (is_alive, _) in results.items()}
            
        finally:
            self._is_checking = False
    
    def check_all_proxies_background(self) -> None:
        """Check proxy trong background thread (non-blocking)"""
        if self._is_checking:
            return
        
        def background_check():
            logger.info("Starting background proxy check...")
            self.check_all_proxies_concurrent()
        
        thread = threading.Thread(target=background_check, daemon=True)
        thread.start()
    
    def get_alive_proxies(self) -> List[ProxyInfo]:
        """Lấy danh sách proxy còn sống"""
        with self._lock:
            alive = [p for p in self.proxies if p.is_alive]
            
            # Nếu lazy_check và không có alive proxy, trigger background check
            if self.lazy_check and not alive and self.proxies:
                logger.info("No alive proxies, triggering background check...")
                self.check_all_proxies_background()
                # Trả về proxy đầu tiên tạm thời (sẽ check trong background)
                if self.proxies:
                    logger.warning("Returning unchecked proxy (background check in progress)")
                    return [self.proxies[0]]
            
            return alive
    
    def get_reliable_proxies(self) -> List[ProxyInfo]:
        """Lấy proxy đáng tin cậy"""
        return [p for p in self.get_alive_proxies() if p.is_reliable]
    
    def get_next_proxy(self, mark_used: bool = True) -> Optional[Dict[str, str]]:
        """Lấy proxy tiếp theo"""
        with self._lock:
            alive_proxies = self.get_alive_proxies()
            
            if not alive_proxies:
                logger.warning("No alive proxies available")
                return None
            
            # Chọn proxy
            selected = None
            
            if self.rotation_strategy == "round_robin":
                self.proxy_index = self.proxy_index % len(alive_proxies)
                selected = alive_proxies[self.proxy_index]
                self.proxy_index += 1
                
            elif self.rotation_strategy == "random":
                selected = random.choice(alive_proxies)
                
            elif self.rotation_strategy == "fastest":
                valid = [p for p in alive_proxies if p.speed is not None]
                if valid:
                    selected = min(valid, key=lambda p: p.speed)
                else:
                    selected = alive_proxies[0]
            
            if selected and mark_used:
                selected.last_used = datetime.now()
                selected.success_count += 1
            
            if selected:
                return {'http': selected.address, 'https': selected.address}
            
            return None
    
    def mark_proxy_failed(self, proxy_dict: Dict[str, str]) -> None:
        """Đánh dấu proxy thất bại"""
        proxy_address = proxy_dict.get('http') or proxy_dict.get('https')
        if not proxy_address:
            return
        
        with self._lock:
            for proxy in self.proxies:
                if proxy.address == proxy_address:
                    proxy.fail_count += 1
                    if proxy.fail_count >= self.max_failures:
                        proxy.is_alive = False
                        logger.debug(f"Proxy {proxy_address} marked DEAD after {proxy.fail_count} failures")
                    break
    
    def mark_proxy_success(self, proxy_dict: Dict[str, str]) -> None:
        """Đánh dấu proxy thành công"""
        proxy_address = proxy_dict.get('http') or proxy_dict.get('https')
        if not proxy_address:
            return
        
        with self._lock:
            for proxy in self.proxies:
                if proxy.address == proxy_address:
                    proxy.success_count += 1
                    proxy.is_alive = True
                    proxy.fail_count = 0
                    break
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê"""
        alive = self.get_alive_proxies()
        return {
            'total_proxies': len(self.proxies),
            'alive_proxies': len(alive),
            'reliable_proxies': len(self.get_reliable_proxies()),
            'rotation_strategy': self.rotation_strategy,
        }


# ============================================
# CÁCH SỬ DỤNG
# ============================================

def demo():
    """Demo cách sử dụng"""
    
    # Cách 1: KHÔNG check khi khởi tạo (nhanh nhất)
    logger.info("=" * 50)
    logger.info("Cách 1: Lazy check - Không bị nghẽn")
    logger.info("=" * 50)
    
    pm1 = ProxyManager(
        auto_fetch_free=True,
        check_on_start=False,      # ✅ Quan trọng: không check
        lazy_check=True,           # ✅ Chỉ check khi cần
        rotation_strategy="random"
    )
    
    # Lấy proxy - lúc này mới check trong background
    proxy = pm1.get_next_proxy()
    if proxy:
        logger.info(f"Got proxy: {proxy['http']}")
    
    # Cách 2: Check concurrent (nhanh hơn nhiều so với tuần tự)
    logger.info("\n" + "=" * 50)
    logger.info("Cách 2: Concurrent check (nếu thực sự cần check)")
    logger.info("=" * 50)
    
    pm2 = ProxyManager(
        auto_fetch_free=True,
        check_on_start=False,
        lazy_check=False
    )
    
    # Chủ động check concurrent
    results = pm2.check_all_proxies_concurrent()
    logger.info(f"Check completed: {sum(results.values())}/{len(results)} alive")
    
    return pm1


if __name__ == "__main__":
    demo()