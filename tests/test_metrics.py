import pytest
from src.utils.metrics import MetricsCollector
import time

@pytest.fixture
def metrics():
    return MetricsCollector()

def test_counter_operations(metrics):
    metrics.increment_counter('test_counter')
    assert metrics.get_counter('test_counter') == 1
    
    metrics.increment_counter('test_counter', 5)
    assert metrics.get_counter('test_counter') == 6

def test_gauge_operations(metrics):
    metrics.set_gauge('test_gauge', 42.5)
    assert metrics.get_gauge('test_gauge') == 42.5
    
    metrics.set_gauge('test_gauge', 10.0)
    assert metrics.get_gauge('test_gauge') == 10.0

def test_histogram_operations(metrics):
    values = [1, 2, 3, 4, 5]
    for value in values:
        metrics.observe_value('test_histogram', value)
    
    stats = metrics.get_histogram_stats('test_histogram')
    assert stats['count'] == 5
    assert stats['mean'] == 3.0
    assert stats['median'] == 3.0
    assert stats['min'] == 1.0
    assert stats['max'] == 5.0

def test_timer_operations(metrics):
    metrics.start_timer('test_operation')
    time.sleep(0.1)  # Simulate some work
    duration = metrics.stop_timer('test_operation')
    
    assert duration >= 0.1
    stats = metrics.get_histogram_stats('test_operation_duration')
    assert stats['count'] == 1
    assert stats['min'] >= 0.1

def test_reset(metrics):
    metrics.increment_counter('test_counter')
    metrics.set_gauge('test_gauge', 42.5)
    metrics.observe_value('test_histogram', 1.0)
    
    metrics.reset()
    
    assert metrics.get_counter('test_counter') == 0
    assert metrics.get_gauge('test_gauge') is None
    assert metrics.get_histogram_stats('test_histogram') == {}