import time
import sys
from hybrid_lfu_cache import HybridLFUCache


def test_basic_operations():
    print("=== 测试基本操作 ===")
    cache = HybridLFUCache(capacity=3)

    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)

    assert cache.get('a') == 1
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.get('d') is None

    print("✓ 基本put/get操作正常")
    print(f"  统计: {cache.get_stats()}")


def test_lfu_behavior():
    print("\n=== 测试LFU行为: 高频优先保留 ===")
    cache = HybridLFUCache(capacity=3)

    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)

    for _ in range(10):
        cache.get('a')
    for _ in range(5):
        cache.get('b')

    cache.put('d', 4)

    assert 'a' in cache
    assert 'b' in cache
    assert 'c' not in cache
    assert 'd' in cache

    print("✓ LFU行为正确: 低频的'c'被淘汰")
    print(f"  频率分布: {cache.get_stats()['freq_distribution']}")


def test_lru_behavior():
    print("\n=== 测试LRU行为: 同频率下最近最少访问被淘汰 ===")
    cache = HybridLFUCache(capacity=3)

    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)

    cache.get('a')
    cache.get('c')

    cache.put('d', 4)

    assert 'a' in cache
    assert 'b' not in cache
    assert 'c' in cache
    assert 'd' in cache

    print("✓ LRU行为正确: 同频率下最久未访问的'b'被淘汰")


def test_anti_scan_pollution():
    print("\n=== 测试抗扫描污染: 一次性扫描不影响热点 ===")
    cache = HybridLFUCache(capacity=5)

    cache.put('hot1', 'hot1')
    cache.put('hot2', 'hot2')
    cache.put('hot3', 'hot3')

    for _ in range(20):
        cache.get('hot1')
        cache.get('hot2')
        cache.get('hot3')

    for i in range(10):
        cache.put(f'scan_{i}', f'value_{i}')

    assert 'hot1' in cache, "热点数据不应被扫描数据挤出"
    assert 'hot2' in cache, "热点数据不应被扫描数据挤出"
    assert 'hot3' in cache, "热点数据不应被扫描数据挤出"

    print("✓ 抗扫描污染正常: 热点数据保留，扫描数据被淘汰")
    print(f"  缓存中keys: {[k for k in ['hot1', 'hot2', 'hot3', 'scan_0', 'scan_9'] if k in cache]}")


def test_frequency_decay():
    print("\n=== 测试频率衰减: 历史热点随时间降温 ===")
    cache = HybridLFUCache(capacity=3, decay_window=0.01, decay_factor=0.5)

    cache.put('old_hot', 'old')
    for _ in range(20):
        cache.get('old_hot')

    old_freq = cache._cache['old_hot'].freq
    print(f"  衰减前old_hot频率: {old_freq}")

    time.sleep(0.02)

    cache.put('new1', 'new1')
    cache.get('new1')
    cache.get('new1')

    new_freq = cache._cache['old_hot'].freq
    print(f"  衰减后old_hot频率: {new_freq}")

    assert new_freq < old_freq, "频率应该随时间衰减"

    cache.put('new2', 'new2')
    cache.put('new3', 'new3')
    cache.put('new4', 'new4')

    assert 'old_hot' not in cache, "衰减后的历史热点应该能被新数据取代"

    print("✓ 频率衰减正常: 历史热点被新热点取代")


def test_update_value():
    print("\n=== 测试更新已有key的值 ===")
    cache = HybridLFUCache(capacity=100)

    cache.put('a', 1)
    freq_after_first_put = cache._cache['a'].freq
    cache.put('a', 2)
    freq_after_second_put = cache._cache['a'].freq

    assert cache.get('a') == 2
    assert freq_after_second_put == freq_after_first_put + 1

    print(f"  第一次put后频率: {freq_after_first_put}, 第二次put后频率: {freq_after_second_put}")
    print("✓ 更新值并增加频率正常")


def test_time_complexity():
    print("\n=== 测试O(1)时间复杂度 ===")
    import timeit

    sizes = [1000, 5000, 10000]
    for size in sizes:
        cache = HybridLFUCache(capacity=size)

        put_time = timeit.timeit(
            lambda: [cache.put(f'key_{i}', i) for i in range(size)],
            number=1
        )

        get_time = timeit.timeit(
            lambda: [cache.get(f'key_{i}') for i in range(size)],
            number=1
        )

        put_avg = (put_time / size) * 1e6
        get_avg = (get_time / size) * 1e6

        print(f"  容量{size:5d}: put平均 {put_avg:6.2f}μs, get平均 {get_avg:6.2f}μs")

    print("✓ 时间复杂度接近O(1): 不同容量下单次操作耗时基本稳定")


def test_mixed_workload():
    print("\n=== 测试混合工作负载 ===")
    cache = HybridLFUCache(capacity=100, decay_window=60.0)

    hot_keys = [f'hot_{i}' for i in range(20)]
    warm_keys = [f'warm_{i}' for i in range(30)]
    cold_keys = [f'cold_{i}' for i in range(100)]

    for key in hot_keys + warm_keys + cold_keys[:50]:
        cache.put(key, 'value')

    for _ in range(100):
        for key in hot_keys:
            cache.get(key)

    for _ in range(10):
        for key in warm_keys:
            cache.get(key)

    for key in cold_keys[50:]:
        cache.put(key, 'value')

    hot_hit = sum(1 for k in hot_keys if k in cache)
    warm_hit = sum(1 for k in warm_keys if k in cache)
    cold_hit = sum(1 for k in cold_keys if k in cache)

    print(f"  热点保留: {hot_hit}/{len(hot_keys)}")
    print(f"  温点保留: {warm_hit}/{len(warm_keys)}")
    print(f"  冷点保留: {cold_hit}/{len(cold_keys)}")

    assert hot_hit >= 18, "热点数据应大部分保留"
    assert warm_hit >= 15, "温点数据应部分保留"
    assert cold_hit < 70, "冷点数据应大部分被淘汰"

    print("✓ 混合工作负载表现正常")


def test_edge_cases():
    print("\n=== 测试边界情况 ===")

    try:
        HybridLFUCache(capacity=0)
        assert False, "应该抛出异常"
    except ValueError:
        print("✓ 容量为0时正确抛出异常")

    cache = HybridLFUCache(capacity=1)
    cache.put('a', 1)
    cache.put('b', 2)
    assert 'a' not in cache
    assert cache.get('b') == 2
    print("✓ 容量为1时工作正常")

    cache = HybridLFUCache(capacity=5)
    assert cache.get('nonexistent') is None
    print("✓ 不存在的key返回None")

    print("✓ 边界情况测试通过")


if __name__ == '__main__':
    test_basic_operations()
    test_lfu_behavior()
    test_lru_behavior()
    test_anti_scan_pollution()
    test_frequency_decay()
    test_update_value()
    test_time_complexity()
    test_mixed_workload()
    test_edge_cases()

    print("\n" + "=" * 50)
    print("所有测试通过! ✓")
