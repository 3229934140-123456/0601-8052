import time
import random
from collections import deque
from typing import Dict, Optional, Any, List, Tuple, Callable


class _DoublyLinkedNode:
    __slots__ = ('key', 'value', 'freq', 'last_access_time', 'prev', 'next')

    def __init__(self, key: Any, value: Any, freq: int = 1):
        self.key = key
        self.value = value
        self.freq = freq
        self.last_access_time = time.monotonic()
        self.prev: Optional[_DoublyLinkedNode] = None
        self.next: Optional[_DoublyLinkedNode] = None


class _DoublyLinkedList:
    def __init__(self):
        self.head = _DoublyLinkedNode(None, None, 0)
        self.tail = _DoublyLinkedNode(None, None, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def add_to_front(self, node: _DoublyLinkedNode) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def add_to_back(self, node: _DoublyLinkedNode) -> None:
        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1

    def remove_node(self, node: _DoublyLinkedNode) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def remove_tail(self) -> Optional[_DoublyLinkedNode]:
        if self.size == 0:
            return None
        node = self.tail.prev
        self.remove_node(node)
        return node

    def is_empty(self) -> bool:
        return self.size == 0


class EvictionRecord:
    def __init__(
        self,
        key: Any,
        freq: int,
        last_access_time: float,
        eviction_time: float,
        key_classifier: Optional[Callable[[Any], str]] = None,
    ):
        self.key = key
        self.freq = freq
        self.last_access_time = last_access_time
        self.eviction_time = eviction_time
        self.key_classifier = key_classifier
        self.category = key_classifier(key) if key_classifier else 'unknown'

    def __repr__(self) -> str:
        return (f"EvictionRecord(key={self.key}, category={self.category}, "
                f"freq={self.freq}, idle={self.eviction_time - self.last_access_time:.3f}s)")


class LatencySample:
    def __init__(self, op_type: str, latency_us: float, result: str):
        self.op_type = op_type
        self.latency_us = latency_us
        self.result = result
        self.timestamp = time.monotonic()

    def __repr__(self) -> str:
        return f"LatencySample(op={self.op_type}, latency={self.latency_us:.2f}μs, result={self.result})"


class HybridLFUCache:
    def __init__(
        self,
        capacity: int,
        decay_window: float = 60.0,
        decay_factor: float = 0.5,
        sample_size: int = 5,
        decay_check_interval: int = 100,
        track_evictions: bool = True,
        max_eviction_records: int = 1000,
        track_latency: bool = True,
        max_latency_samples: int = 10000,
        key_classifier: Optional[Callable[[Any], str]] = None,
    ):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if sample_size <= 0:
            raise ValueError("Sample size must be positive")

        self.capacity = capacity
        self.decay_window = decay_window
        self.decay_factor = decay_factor
        self.sample_size = sample_size
        self.decay_check_interval = decay_check_interval

        self._cache: Dict[Any, _DoublyLinkedNode] = {}
        self._freq_buckets: Dict[int, _DoublyLinkedList] = {}
        self._min_freq = 1
        self._access_count = 0

        self._all_keys: List[Any] = []
        self._key_index: Dict[Any, int] = {}

        self.eviction_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.decay_count = 0

        self.track_evictions = track_evictions
        self.key_classifier = key_classifier
        self._eviction_records: deque = deque(maxlen=max_eviction_records) if track_evictions else None
        self._eviction_by_category: Dict[str, int] = {}

        self.track_latency = track_latency
        self._latency_samples: deque = deque(maxlen=max_latency_samples) if track_latency else None
        self._op_count: Dict[str, int] = {'get': 0, 'put': 0, 'evict': 0}

    def _classify_key(self, key: Any) -> str:
        if self.key_classifier:
            return self.key_classifier(key)
        return 'unknown'

    def _record_eviction(self, node: _DoublyLinkedNode) -> None:
        if not self.track_evictions:
            return

        record = EvictionRecord(
            key=node.key,
            freq=node.freq,
            last_access_time=node.last_access_time,
            eviction_time=time.monotonic(),
            key_classifier=self.key_classifier,
        )
        self._eviction_records.append(record)
        category = record.category
        self._eviction_by_category[category] = self._eviction_by_category.get(category, 0) + 1

    def _record_latency(self, op_type: str, latency_us: float, result: str) -> None:
        if not self.track_latency:
            return
        self._latency_samples.append(LatencySample(op_type, latency_us, result))
        self._op_count[op_type] = self._op_count.get(op_type, 0) + 1

    def _get_or_create_bucket(self, freq: int) -> _DoublyLinkedList:
        if freq not in self._freq_buckets:
            self._freq_buckets[freq] = _DoublyLinkedList()
        return self._freq_buckets[freq]

    def _add_key_to_index(self, key: Any) -> None:
        self._all_keys.append(key)
        self._key_index[key] = len(self._all_keys) - 1

    def _remove_key_from_index(self, key: Any) -> None:
        if key not in self._key_index:
            return
        idx = self._key_index[key]
        last_key = self._all_keys[-1]
        self._all_keys[idx] = last_key
        self._key_index[last_key] = idx
        self._all_keys.pop()
        del self._key_index[key]

    def _random_sample_keys(self, n: int) -> List[Any]:
        size = len(self._all_keys)
        if size == 0:
            return []
        n = min(n, size)
        indices = random.sample(range(size), n)
        return [self._all_keys[i] for i in indices]

    def _try_decay_node(self, node: _DoublyLinkedNode, current_time: float) -> bool:
        if node.freq <= 1:
            return False

        time_since_access = current_time - node.last_access_time
        if time_since_access <= self.decay_window:
            return False

        old_freq = node.freq
        new_freq = max(1, int(old_freq * self.decay_factor))

        if new_freq >= old_freq:
            return False

        old_bucket = self._freq_buckets[old_freq]
        old_bucket.remove_node(node)

        if old_bucket.is_empty():
            del self._freq_buckets[old_freq]
            if old_freq == self._min_freq:
                if self._freq_buckets:
                    self._min_freq = min(self._freq_buckets.keys())
                else:
                    self._min_freq = 1

        node.freq = new_freq
        new_bucket = self._get_or_create_bucket(new_freq)
        new_bucket.add_to_back(node)

        self.decay_count += 1
        return True

    def _decay_if_needed(self) -> None:
        self._access_count += 1
        if self._access_count % self.decay_check_interval != 0:
            return

        current_time = time.monotonic()
        sample_keys = self._random_sample_keys(self.sample_size)

        for key in sample_keys:
            if key in self._cache:
                node = self._cache[key]
                self._try_decay_node(node, current_time)

    def _bump_node_freq(self, node: _DoublyLinkedNode) -> None:
        old_freq = node.freq
        old_bucket = self._freq_buckets[old_freq]
        old_bucket.remove_node(node)

        if old_bucket.is_empty():
            del self._freq_buckets[old_freq]
            if old_freq == self._min_freq:
                self._min_freq = old_freq + 1

        node.freq += 1
        node.last_access_time = time.monotonic()

        new_bucket = self._get_or_create_bucket(node.freq)
        new_bucket.add_to_front(node)

    def get(self, key: Any) -> Optional[Any]:
        t0 = time.perf_counter()

        if key not in self._cache:
            self.miss_count += 1
            t1 = time.perf_counter()
            self._record_latency('get', (t1 - t0) * 1e6, 'miss')
            return None

        self.hit_count += 1
        node = self._cache[key]

        current_time = time.monotonic()
        self._try_decay_node(node, current_time)

        self._bump_node_freq(node)

        self._decay_if_needed()

        t1 = time.perf_counter()
        self._record_latency('get', (t1 - t0) * 1e6, 'hit')

        return node.value

    def put(self, key: Any, value: Any) -> None:
        t0 = time.perf_counter()

        if self.capacity == 0:
            t1 = time.perf_counter()
            self._record_latency('put', (t1 - t0) * 1e6, 'skipped')
            return

        if key in self._cache:
            node = self._cache[key]
            node.value = value

            current_time = time.monotonic()
            self._try_decay_node(node, current_time)

            self._bump_node_freq(node)

            self._decay_if_needed()

            t1 = time.perf_counter()
            self._record_latency('put', (t1 - t0) * 1e6, 'update')
            return

        if len(self._cache) >= self.capacity:
            self._evict()

        node = _DoublyLinkedNode(key, value, freq=1)
        self._cache[key] = node
        self._add_key_to_index(key)

        bucket = self._get_or_create_bucket(1)
        bucket.add_to_front(node)
        self._min_freq = 1

        self._decay_if_needed()

        t1 = time.perf_counter()
        self._record_latency('put', (t1 - t0) * 1e6, 'insert')

    def _evict(self) -> None:
        t0 = time.perf_counter()

        bucket = self._freq_buckets[self._min_freq]
        evicted_node = bucket.remove_tail()

        if evicted_node and evicted_node.key is not None:
            key = evicted_node.key
            self._record_eviction(evicted_node)
            del self._cache[key]
            self._remove_key_from_index(key)
            self.eviction_count += 1

        if bucket.is_empty():
            del self._freq_buckets[self._min_freq]
            if self._freq_buckets:
                self._min_freq = min(self._freq_buckets.keys())
            else:
                self._min_freq = 1

        t1 = time.perf_counter()
        self._record_latency('evict', (t1 - t0) * 1e6, 'evicted')

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def get_recent_evictions(self, n: int = 10) -> List[EvictionRecord]:
        if not self.track_evictions or not self._eviction_records:
            return []
        return list(self._eviction_records)[-n:]

    def get_eviction_by_category(self) -> Dict[str, int]:
        return dict(self._eviction_by_category)

    def get_latency_percentiles(self, op_type: Optional[str] = None) -> Dict[str, float]:
        if not self.track_latency or not self._latency_samples:
            return {}

        if op_type:
            samples = [s.latency_us for s in self._latency_samples if s.op_type == op_type]
        else:
            samples = [s.latency_us for s in self._latency_samples]

        if not samples:
            return {}

        samples.sort()
        n = len(samples)
        return {
            'count': n,
            'min': round(samples[0], 2),
            'p50': round(samples[int(n * 0.5)], 2),
            'p90': round(samples[int(n * 0.9)], 2),
            'p95': round(samples[int(n * 0.95)], 2),
            'p99': round(samples[int(n * 0.99)], 2),
            'max': round(samples[-1], 2),
            'avg': round(sum(samples) / n, 2),
        }

    def get_latency_distribution(self, op_type: Optional[str] = None) -> Dict[str, int]:
        if not self.track_latency or not self._latency_samples:
            return {}

        if op_type:
            samples = [s.latency_us for s in self._latency_samples if s.op_type == op_type]
        else:
            samples = [s.latency_us for s in self._latency_samples]

        if not samples:
            return {}

        buckets = [
            ('<1μs', 0, 1),
            ('1-5μs', 1, 5),
            ('5-10μs', 5, 10),
            ('10-50μs', 10, 50),
            ('50-100μs', 50, 100),
            ('>100μs', 100, float('inf')),
        ]

        result = {}
        for label, lo, hi in buckets:
            count = sum(1 for s in samples if lo <= s < hi)
            if count > 0:
                result[label] = count

        return result

    def get_stats(self) -> Dict[str, Any]:
        freq_distribution = {
            freq: bucket.size
            for freq, bucket in self._freq_buckets.items()
            if not bucket.is_empty()
        }
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0

        result = {
            'size': len(self._cache),
            'capacity': self.capacity,
            'min_freq': self._min_freq,
            'freq_distribution': freq_distribution,
            'decay_window': self.decay_window,
            'decay_factor': self.decay_factor,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': round(hit_rate, 2),
            'eviction_count': self.eviction_count,
            'decay_count': self.decay_count,
        }

        if self.track_evictions:
            result['eviction_by_category'] = self.get_eviction_by_category()
            result['recent_evictions_count'] = len(self._eviction_records) if self._eviction_records else 0

        if self.track_latency:
            result['latency_p50_get'] = self.get_latency_percentiles('get').get('p50', 0)
            result['latency_p99_get'] = self.get_latency_percentiles('get').get('p99', 0)

        return result

    def get_detailed_stats(self) -> Dict[str, Any]:
        stats = self.get_stats()

        if self.track_latency:
            stats['latency_get'] = self.get_latency_percentiles('get')
            stats['latency_put'] = self.get_latency_percentiles('put')
            stats['latency_all'] = self.get_latency_percentiles()
            stats['latency_distribution'] = self.get_latency_distribution()

        if self.track_evictions:
            stats['recent_evictions'] = self.get_recent_evictions(20)

        return stats

    def reset_stats(self) -> None:
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.decay_count = 0
        self._access_count = 0
        self._eviction_by_category.clear()
        if self._eviction_records:
            self._eviction_records.clear()
        if self._latency_samples:
            self._latency_samples.clear()
        self._op_count = {'get': 0, 'put': 0, 'evict': 0}
