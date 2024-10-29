from typing import Dict, Union, Optional
from datetime import datetime
from collections import defaultdict
from src.utils.logger import get_main_logger, get_rag_logger

logger = get_main_logger()
rag_logger = get_rag_logger()

class MetricsCollector:
    """Collector for application metrics with support for counters, gauges, and histograms."""
    
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)
        self.start_times: Dict[str, datetime] = {}
        logger.info("MetricsCollector initialized")
        rag_logger.info("\nMetrics System:\nStatus: Initialized\n" + "-"*50)
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] += value
        logger.debug(f"Counter {name} incremented by {value}")
        rag_logger.debug(f"\nMetric Update:\nCounter: {name}\nValue: +{value}\n{'-'*50}")
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric to a specific value."""
        self.gauges[name] = value
        logger.debug(f"Gauge {name} set to {value}")
        rag_logger.debug(f"\nMetric Update:\nGauge: {name}\nValue: {value}\n{'-'*50}")
    
    def observe_value(self, name: str, value: Union[int, float]) -> None:
        """Record a value in a histogram metric."""
        self.histograms[name].append(value)
        logger.debug(f"Observed value {value} for {name}")
        rag_logger.debug(f"\nMetric Update:\nHistogram: {name}\nValue: {value}\n{'-'*50}")
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation."""
        self.start_times[name] = datetime.now()
        logger.debug(f"Timer {name} started")
        rag_logger.debug(f"\nTimer Start:\nOperation: {name}\n{'-'*50}")
    
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop timing an operation and record the duration."""
        if name not in self.start_times:
            error_msg = f"Timer {name} was never started"
            logger.warning(error_msg)
            rag_logger.warning(f"\nTimer Error:\n{error_msg}\n{'-'*50}")
            return None
            
        duration = (datetime.now() - self.start_times[name]).total_seconds()
        self.observe_value(f"{name}_duration", duration)
        del self.start_times[name]
        
        logger.debug(f"Timer {name} stopped, duration: {duration:.3f}s")
        rag_logger.debug(
            f"\nTimer Complete:\n"
            f"Operation: {name}\n"
            f"Duration: {duration:.3f}s\n"
            f"{'-'*50}"
        )
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
            logger.debug(f"No values found for histogram {name}")
            return {}
            
        import numpy as np
        stats = {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
        logger.debug(f"Generated stats for histogram {name}")
        rag_logger.debug(
            f"\nHistogram Stats:\n"
            f"Metric: {name}\n"
            f"Count: {stats['count']}\n"
            f"Mean: {stats['mean']:.3f}\n"
            f"Median: {stats['median']:.3f}\n"
            f"Min: {stats['min']:.3f}\n"
            f"Max: {stats['max']:.3f}\n"
            f"P95: {stats['p95']:.3f}\n"
            f"P99: {stats['p99']:.3f}\n"
            f"{'-'*50}"
        )
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.start_times.clear()
        logger.info("All metrics reset")
        rag_logger.info("\nMetrics Reset:\nStatus: Completed\n" + "-"*50)