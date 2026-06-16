import time
import random
from typing import Dict, Optional, Any, List


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


class HybridLFUCache:
    def __init__(
        self,
        capacity: int,
        decay_window: float = 60.0,
        decay_factor: float = 0.5,
        sample_size: int = 5,
        decay_check_interval: int = 100,
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
        if key not in self._cache:
            self.miss_count += 1
            return None

        self.hit_count += 1
        node = self._cache[key]

        current_time = time.monotonic()
        self._try_decay_node(node, current_time)

        self._bump_node_freq(node)

        self._decay_if_needed()

        return node.value

    def put(self, key: Any, value: Any) -> None:
        if self.capacity == 0:
            return

        if key in self._cache:
            node = self._cache[key]
            node.value = value

            current_time = time.monotonic()
            self._try_decay_node(node, current_time)

            self._bump_node_freq(node)

            self._decay_if_needed()
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

    def _evict(self) -> None:
        bucket = self._freq_buckets[self._min_freq]
        evicted_node = bucket.remove_tail()

        if evicted_node and evicted_node.key is not None:
            key = evicted_node.key
            del self._cache[key]
            self._remove_key_from_index(key)
            self.eviction_count += 1

        if bucket.is_empty():
            del self._freq_buckets[self._min_freq]
            if self._freq_buckets:
                self._min_freq = min(self._freq_buckets.keys())
            else:
                self._min_freq = 1

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def get_stats(self) -> Dict[str, Any]:
        freq_distribution = {
            freq: bucket.size
            for freq, bucket in self._freq_buckets.items()
            if not bucket.is_empty()
        }
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0

        return {
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

    def reset_stats(self) -> None:
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.decay_count = 0
