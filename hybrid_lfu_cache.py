import time
from typing import Dict, Optional, Any


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
    def __init__(self, capacity: int, decay_window: float = 60.0, decay_factor: float = 0.5):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.decay_window = decay_window
        self.decay_factor = decay_factor

        self._cache: Dict[Any, _DoublyLinkedNode] = {}
        self._freq_buckets: Dict[int, _DoublyLinkedList] = {}
        self._min_freq = 1
        self._access_count = 0
        self._decay_check_interval = max(1, capacity // 10)

    def _get_or_create_bucket(self, freq: int) -> _DoublyLinkedList:
        if freq not in self._freq_buckets:
            self._freq_buckets[freq] = _DoublyLinkedList()
        return self._freq_buckets[freq]

    def _decay_if_needed(self) -> None:
        self._access_count += 1
        if self._access_count % self._decay_check_interval != 0:
            return

        current_time = time.monotonic()
        nodes_to_decay = []

        for freq in list(self._freq_buckets.keys()):
            if freq <= 1:
                continue
            bucket = self._freq_buckets[freq]
            if bucket.is_empty():
                continue

            node = bucket.head.next
            while node != bucket.tail:
                next_node = node.next
                if current_time - node.last_access_time > self.decay_window:
                    nodes_to_decay.append(node)
                node = next_node

        for node in nodes_to_decay:
            old_freq = node.freq
            new_freq = max(1, int(old_freq * self.decay_factor))

            if new_freq != old_freq:
                old_bucket = self._freq_buckets[old_freq]
                old_bucket.remove_node(node)

                if old_bucket.is_empty() and old_freq == self._min_freq:
                    self._min_freq = new_freq

                node.freq = new_freq
                new_bucket = self._get_or_create_bucket(new_freq)
                new_bucket.add_to_front(node)

        for freq in list(self._freq_buckets.keys()):
            if self._freq_buckets[freq].is_empty():
                del self._freq_buckets[freq]

        if not self._freq_buckets:
            self._min_freq = 1
        else:
            self._min_freq = min(self._freq_buckets.keys())

    def get(self, key: Any) -> Optional[Any]:
        if key not in self._cache:
            return None

        node = self._cache[key]

        old_freq = node.freq
        old_bucket = self._freq_buckets[old_freq]
        old_bucket.remove_node(node)

        if old_bucket.is_empty() and old_freq == self._min_freq:
            self._min_freq = old_freq + 1

        node.freq += 1
        node.last_access_time = time.monotonic()

        new_bucket = self._get_or_create_bucket(node.freq)
        new_bucket.add_to_front(node)

        self._decay_if_needed()

        return node.value

    def put(self, key: Any, value: Any) -> None:
        if self.capacity == 0:
            return

        if key in self._cache:
            node = self._cache[key]
            node.value = value

            old_freq = node.freq
            old_bucket = self._freq_buckets[old_freq]
            old_bucket.remove_node(node)

            if old_bucket.is_empty() and old_freq == self._min_freq:
                self._min_freq = old_freq + 1

            node.freq += 1
            node.last_access_time = time.monotonic()

            new_bucket = self._get_or_create_bucket(node.freq)
            new_bucket.add_to_front(node)

            self._decay_if_needed()
            return

        if len(self._cache) >= self.capacity:
            self._evict()

        node = _DoublyLinkedNode(key, value, freq=1)
        self._cache[key] = node

        bucket = self._get_or_create_bucket(1)
        bucket.add_to_front(node)
        self._min_freq = 1

        self._decay_if_needed()

    def _evict(self) -> None:
        bucket = self._freq_buckets[self._min_freq]
        evicted_node = bucket.remove_tail()

        if evicted_node and evicted_node.key in self._cache:
            del self._cache[evicted_node.key]

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
        return {
            'size': len(self._cache),
            'capacity': self.capacity,
            'min_freq': self._min_freq,
            'freq_distribution': freq_distribution,
            'decay_window': self.decay_window,
            'decay_factor': self.decay_factor,
        }
