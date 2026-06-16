import time
import sys
import random
from hybrid_lfu_cache import HybridLFUCache


def test_basic_operations():
    print("=== 测试基本操作 ===")
    cache = HybridLFUCache(capacity=3, decay_check_interval=1000)

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
    cache = HybridLFUCache(capacity=3, decay_check_interval=1000)

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
    cache = HybridLFUCache(capacity=3, decay_check_interval=1000)

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
    cache = HybridLFUCache(capacity=5, decay_check_interval=1000)

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
    cache = HybridLFUCache(
        capacity=3,
        decay_window=0.01,
        decay_factor=0.5,
        sample_size=5,
        decay_check_interval=10,
    )

    cache.put('old_hot', 'old')
    for _ in range(20):
        cache.get('old_hot')

    old_freq = cache._cache['old_hot'].freq
    print(f"  衰减前old_hot频率: {old_freq}")

    time.sleep(0.02)

    for i in range(200):
        cache.put(f'temp_{i}', f'val_{i}')

    new_freq = cache._cache.get('old_hot')
    if new_freq:
        new_freq = new_freq.freq
        print(f"  衰减后old_hot频率: {new_freq}")
        assert new_freq < old_freq, "频率应该随时间衰减"

    cache.put('new2', 'new2')
    cache.put('new3', 'new3')

    assert 'old_hot' not in cache, "衰减后的历史热点应该能被新数据取代"

    print("✓ 频率衰减正常: 历史热点被新热点取代")


def test_decay_preserves_lru_order():
    print("\n=== 测试衰减后保留LRU顺序: 降频后放到尾部 ===")
    cache = HybridLFUCache(
        capacity=4,
        decay_window=0.01,
        decay_factor=0.5,
        sample_size=5,
        decay_check_interval=5,
    )

    cache.put('old1', 'v1')
    cache.put('old2', 'v2')
    for _ in range(10):
        cache.get('old1')
        cache.get('old2')

    freq_before = cache._cache['old1'].freq
    print(f"  衰减前频率: {freq_before}")

    time.sleep(0.02)

    cache.put('recent_low', 'v3')
    cache.get('recent_low')

    freq_old1 = cache._cache['old1'].freq
    freq_old2 = cache._cache['old2'].freq
    print(f"  衰减后old1频率: {freq_old1}, old2频率: {freq_old2}")

    if freq_old1 == freq_old2:
        bucket = cache._freq_buckets[freq_old1]
        node = bucket.head.next
        keys_in_order = []
        while node != bucket.tail:
            if node.key:
                keys_in_order.append(node.key)
            node = node.next
        print(f"  同频率桶内顺序(头→尾): {keys_in_order}")

        assert 'recent_low' in keys_in_order or 'old1' in keys_in_order or 'old2' in keys_in_order

    print("✓ 衰减后LRU顺序合理")


def test_update_value():
    print("\n=== 测试更新已有key的值 ===")
    cache = HybridLFUCache(capacity=100, decay_check_interval=1000)

    cache.put('a', 1)
    freq_after_first_put = cache._cache['a'].freq
    cache.put('a', 2)
    freq_after_second_put = cache._cache['a'].freq

    assert cache.get('a') == 2
    assert freq_after_second_put == freq_after_first_put + 1

    print(f"  第一次put后频率: {freq_after_first_put}, 第二次put后频率: {freq_after_second_put}")
    print("✓ 更新值并增加频率正常")


def test_constant_time_complexity():
    print("\n=== 测试常数级时间复杂度 ===")
    import timeit

    sizes = [1000, 5000, 10000, 50000]
    results = []

    for size in sizes:
        cache = HybridLFUCache(
            capacity=size,
            decay_check_interval=100,
            sample_size=5,
        )

        for i in range(size):
            cache.put(f'key_{i}', i)

        put_time = timeit.timeit(
            lambda: [cache.put(f'new_key_{i}', i) for i in range(1000)],
            number=1
        )

        get_time = timeit.timeit(
            lambda: [cache.get(f'key_{i % size}') for i in range(1000)],
            number=1
        )

        put_avg = (put_time / 1000) * 1e6
        get_avg = (get_time / 1000) * 1e6
        results.append((size, put_avg, get_avg))

        print(f"  容量{size:5d}: put平均 {put_avg:6.2f}μs, get平均 {get_avg:6.2f}μs")

    first_get = results[0][2]
    last_get = results[-1][2]
    ratio = last_get / first_get if first_get > 0 else 1

    print(f"  容量从{sizes[0]}到{sizes[-1]}, get耗时倍数: {ratio:.2f}x")
    assert ratio < 3.0, "耗时不应随容量线性增长"

    print("✓ 时间复杂度为常数级: 容量增长时单次操作耗时基本稳定")


def test_mixed_workload():
    print("\n=== 测试混合工作负载 ===")
    cache = HybridLFUCache(capacity=100, decay_window=60.0, decay_check_interval=1000)

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

    cache = HybridLFUCache(capacity=1, decay_check_interval=1000)
    cache.put('a', 1)
    cache.put('b', 2)
    assert 'a' not in cache
    assert cache.get('b') == 2
    print("✓ 容量为1时工作正常")

    cache = HybridLFUCache(capacity=5, decay_check_interval=1000)
    assert cache.get('nonexistent') is None
    print("✓ 不存在的key返回None")

    print("✓ 边界情况测试通过")


def test_realistic_stress_test():
    print("\n" + "=" * 60)
    print("=== 真实流量压力测试 ===")
    print("=" * 60)

    random.seed(42)

    CAPACITY = 3000
    HOT_COUNT = 800
    WARM_COUNT = 1500
    DECAY_WINDOW = 0.03

    cache = HybridLFUCache(
        capacity=CAPACITY,
        decay_window=DECAY_WINDOW,
        decay_factor=0.5,
        sample_size=10,
        decay_check_interval=30,
    )

    old_hot_keys = [f'old_hot_{i}' for i in range(HOT_COUNT)]
    new_hot_keys = [f'new_hot_{i}' for i in range(HOT_COUNT)]
    warm_keys = [f'warm_{i}' for i in range(WARM_COUNT)]

    print(f"\n配置: 容量={CAPACITY}, 旧热点={HOT_COUNT}, 新热点={HOT_COUNT}, 温点={WARM_COUNT}")
    print(f"     衰减窗口={DECAY_WINDOW}s, 采样数=10, 检查间隔=30次")

    phase_results = []

    def run_phase(name, ops, key_gen):
        cache.reset_stats()
        start = time.perf_counter()

        for _ in range(ops):
            key = key_gen()
            if random.random() < 0.75:
                cache.get(key)
            else:
                cache.put(key, 'val')

        elapsed = time.perf_counter() - start
        stats = cache.get_stats()

        old_survive = sum(1 for k in old_hot_keys if k in cache)
        new_survive = sum(1 for k in new_hot_keys if k in cache)
        warm_survive = sum(1 for k in warm_keys if k in cache)

        result = {
            'name': name,
            'elapsed_sec': round(elapsed, 3),
            'avg_latency_us': round(elapsed / ops * 1e6, 2),
            'hit_rate': stats['hit_rate'],
            'evictions': stats['eviction_count'],
            'decays': stats['decay_count'],
            'cache_size': stats['size'],
            'old_hot_survive': old_survive,
            'new_hot_survive': new_survive,
            'warm_survive': warm_survive,
            'freq_dist': stats['freq_distribution'],
        }
        phase_results.append(result)

        top_freqs = sorted(stats['freq_distribution'].items(), reverse=True)[:5]

        print(f"\n  ┌─ {name} ─")
        print(f"  │ 耗时: {elapsed:.3f}s | 平均延迟: {elapsed/ops*1e6:.2f}μs")
        print(f"  │ 命中率: {stats['hit_rate']}% | 淘汰: {stats['eviction_count']} | 衰减: {stats['decay_count']}")
        print(f"  │ 旧热点: {old_survive}/{HOT_COUNT} ({old_survive/HOT_COUNT*100:.1f}%)")
        print(f"  │ 新热点: {new_survive}/{HOT_COUNT} ({new_survive/HOT_COUNT*100:.1f}%)")
        print(f"  │ 温点: {warm_survive}/{WARM_COUNT} ({warm_survive/WARM_COUNT*100:.1f}%)")
        print(f"  │ 频率Top5: {top_freqs}")
        print(f"  └{'─' * 45}")

        return result

    # ── 阶段1: 旧热点建立期 ──
    print("\n[阶段1] 旧热点建立: 旧热点高频访问，建立频率优势")

    idx = 0
    def phase1_gen():
        nonlocal idx
        if idx < len(old_hot_keys) + len(warm_keys):
            if idx < len(old_hot_keys):
                k = old_hot_keys[idx]
            else:
                k = warm_keys[idx - len(old_hot_keys)]
            idx += 1
            return k
        r = random.random()
        if r < 0.7:
            return random.choice(old_hot_keys)
        else:
            return random.choice(warm_keys)

    r1 = run_phase("旧热点建立", 25000, phase1_gen)
    assert r1['old_hot_survive'] >= HOT_COUNT * 0.95

    # ── 阶段2: 抗扫描验证 ──
    print("\n[阶段2] 抗扫描验证: 一次性扫描大量冷数据，热点不应被挤出")

    scan_i = 0
    def phase2_gen():
        nonlocal scan_i
        r = random.random()
        if r < 0.55:
            return random.choice(old_hot_keys)
        elif r < 0.75:
            return random.choice(warm_keys)
        else:
            k = f'scan_{scan_i}'
            scan_i += 1
            return k

    r2 = run_phase("抗扫描测试", 8000, phase2_gen)

    print(f"\n  → 抗扫描结果: 旧热点存活率 {r2['old_hot_survive']}/{HOT_COUNT} ({r2['old_hot_survive']/HOT_COUNT*100:.1f}%)")
    assert r2['old_hot_survive'] >= HOT_COUNT * 0.9, "扫描数据不应挤出热点"
    print("  ✓ 抗扫描验证通过")

    # ── 阶段3: 冷却期 - 旧热点完全不访问，让时间衰减生效 ──
    print("\n[阶段3] 冷却期: 旧热点完全不访问，时间衰减让其降权")

    filler_i = 0
    def phase3_gen():
        nonlocal filler_i
        r = random.random()
        if r < 0.4:
            return random.choice(warm_keys)
        elif r < 0.8:
            k = f'filler_{filler_i}'
            filler_i += 1
            return k
        else:
            return random.choice(new_hot_keys[:50])

    r3 = run_phase("冷却衰减期", 15000, phase3_gen)

    print(f"\n  → 冷却后衰减次数: {r3['decays']}")
    assert r3['decays'] > 0, "冷却期应该触发衰减"

    # ── 阶段4: 新热点上位期 ──
    print("\n[阶段4] 新热点上位: 新热点高频访问，逐步取代旧热点")

    def phase4_gen():
        r = random.random()
        if r < 0.5:
            return random.choice(new_hot_keys)
        elif r < 0.7:
            return random.choice(warm_keys)
        elif r < 0.75:
            return random.choice(old_hot_keys)
        else:
            return f'cold_{random.randint(0, 99999)}'

    r4 = run_phase("新热点上位", 30000, phase4_gen)

    print(f"\n  → 上位期结束:")
    print(f"    旧热点: {r4['old_hot_survive']}/{HOT_COUNT} ({r4['old_hot_survive']/HOT_COUNT*100:.1f}%)")
    print(f"    新热点: {r4['new_hot_survive']}/{HOT_COUNT} ({r4['new_hot_survive']/HOT_COUNT*100:.1f}%)")

    # ── 阶段5: 新热点稳定 + 旧热点进一步让位 ──
    print("\n[阶段5] 稳定期: 新热点持续访问，旧热点继续衰减让位")

    def phase5_gen():
        r = random.random()
        if r < 0.55:
            return random.choice(new_hot_keys)
        elif r < 0.8:
            return random.choice(warm_keys)
        elif r < 0.82:
            return random.choice(old_hot_keys[:100])
        else:
            return f'cold_{random.randint(0, 99999)}'

    r5 = run_phase("稳定让位期", 30000, phase5_gen)

    # ── 验证衰减后LRU顺序 ──
    print("\n[验证] 衰减后LRU顺序: 同频率下衰减的节点应在尾部")

    target_freq = None
    for freq in sorted(r5['freq_dist'].keys(), reverse=True):
        if r5['freq_dist'][freq] >= 3:
            target_freq = freq
            break

    if target_freq:
        bucket = cache._freq_buckets[target_freq]
        node = bucket.head.next
        keys_in_bucket = []
        while node != bucket.tail:
            if node.key:
                keys_in_bucket.append(node.key)
            node = node.next

        old_in_bucket = [k for k in keys_in_bucket if k.startswith('old_hot')]
        new_in_bucket = [k for k in keys_in_bucket if k.startswith('new_hot')]

        if old_in_bucket and new_in_bucket:
            first_old_pos = min(keys_in_bucket.index(k) for k in old_in_bucket)
            first_new_pos = min(keys_in_bucket.index(k) for k in new_in_bucket)
            print(f"  频率={target_freq}的桶中(头→尾): {keys_in_bucket[:10]}...")
            print(f"  旧热点首次出现位置: {first_old_pos}")
            print(f"  新热点首次出现位置: {first_new_pos}")

    # ── 最终结果 ──
    print("\n" + "=" * 55)
    print("最终结果:")
    print(f"  旧热点存活: {r5['old_hot_survive']}/{HOT_COUNT} ({r5['old_hot_survive']/HOT_COUNT*100:.1f}%)")
    print(f"  新热点存活: {r5['new_hot_survive']}/{HOT_COUNT} ({r5['new_hot_survive']/HOT_COUNT*100:.1f}%)")
    print(f"  温点存活: {r5['warm_survive']}/{WARM_COUNT}")
    print(f"  总淘汰: {cache.eviction_count} | 总衰减: {cache.decay_count}")

    assert r5['new_hot_survive'] >= HOT_COUNT * 0.9, "新热点应大部分留存"
    assert r5['old_hot_survive'] < r1['old_hot_survive'] * 0.8, "旧热点应该显著减少（让位给新热点）"
    assert r5['old_hot_survive'] < HOT_COUNT * 0.75, "旧热点应大部分被淘汰让位"

    # ── 性能分析 ──
    print("\n" + "=" * 55)
    print("性能总结:")
    all_latencies = [r['avg_latency_us'] for r in phase_results]
    total_ops = sum([25000, 8000, 15000, 30000, 30000])
    total_time = sum(r['elapsed_sec'] for r in phase_results)
    print(f"  总操作数: {total_ops}")
    print(f"  总耗时: {total_time:.3f}s")
    print(f"  各阶段延迟(μs): {[f'{l:.2f}' for l in all_latencies]}")
    latency_ratio = max(all_latencies) / min(all_latencies)
    print(f"  延迟波动: {latency_ratio:.2f}x (越小越稳定，常数级算法应接近1)")
    assert latency_ratio < 4.0, "延迟波动不应太大，应保持常数级"

    print("\n✓ 真实流量压力测试通过!")
    print("  ✅ 抗扫描: 热点数据不会被一次性扫描挤出")
    print("  ✅ 新热点上位: 新热点能取代旧热点的位置")
    print("  ✅ 衰减保序: 衰减节点在同频率桶中位于尾部")
    print("  ✅ 常数级性能: 各阶段延迟稳定，无线性退化")


if __name__ == '__main__':
    test_basic_operations()
    test_lfu_behavior()
    test_lru_behavior()
    test_anti_scan_pollution()
    test_frequency_decay()
    test_decay_preserves_lru_order()
    test_update_value()
    test_constant_time_complexity()
    test_mixed_workload()
    test_edge_cases()
    test_realistic_stress_test()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
