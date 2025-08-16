# Adaptive Caching System
import functools
import time
from typing import Any, Dict, Optional

class AdaptiveCacheManager:
    """Adaptive caching with learning patterns."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.access_patterns: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.01
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with pattern learning."""
        if key in self.cache:
            self._update_access_pattern(key, hit=True)
            return self.cache[key]
        else:
            self._update_access_pattern(key, hit=False)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache."""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl,
            'access_count': 0
        }
    
    def _update_access_pattern(self, key: str, hit: bool) -> None:
        """Update access patterns for adaptive optimization."""
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'frequency': 0.0,
                'recency': time.time(),
                'hit_rate': 0.0
            }
        
        pattern = self.access_patterns[key]
        pattern['frequency'] = pattern['frequency'] * (1 - self.learning_rate) + self.learning_rate
        pattern['recency'] = time.time()
        if hit:
            pattern['hit_rate'] = pattern['hit_rate'] * (1 - self.learning_rate) + self.learning_rate
    
    def optimize_cache(self) -> None:
        """Optimize cache based on learned patterns."""
        # Remove items with low access patterns
        current_time = time.time()
        to_remove = []
        
        for key, pattern in self.access_patterns.items():
            if (current_time - pattern['recency'] > 3600 and 
                pattern['frequency'] < 0.1 and 
                pattern['hit_rate'] < 0.3):
                to_remove.append(key)
        
        for key in to_remove:
            if key in self.cache:
                del self.cache[key]
            del self.access_patterns[key]

# Global adaptive cache instance
adaptive_cache = AdaptiveCacheManager()