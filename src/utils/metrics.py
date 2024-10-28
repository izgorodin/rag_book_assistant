from typing import Dict, Union, Optional
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collector for application metrics with support for counters, gauges, and histograms."""
    
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)
        self.start_times: Dict[str, datetime] = {}
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] += value
        logger.debug(f"Counter {name} incremented by {value}")
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric to a specific value."""
        self.gauges[name] = value
        logger.debug(f"Gauge {name} set to {value}")
    
    def observe_value(self, name: str, value: Union[int, float]) -> None:
        """Record a value in a histogram metric."""
        self.histograms[name].append(value)
        logger.debug(f"Observed value {value} for {name}")
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation."""
        self.start_times[name] = datetime.now()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop timing an operation and record the duration."""
        if name not in self.start_times:
            logger.warning(f"Timer {name} was never started")
            return None
            
        duration = (datetime.now() - self.start_times[name]).total_seconds()
        self.observe_value(f"{name}_duration", duration)
        del self.start_times[name]
        return duration
    
    def get_counter(self, name: str) -> int:
        """Get the current value of a counter."""
        return self.counters[name]
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get the current value of a gauge."""
        return self.gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of a histogram."""
        values = self.histograms.get(name, [])
        if not values:
            return {}
            
        import numpy as np
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.start_times.clear()