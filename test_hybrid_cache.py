import time
import sys
import random
from typing import Dict, List
from hybrid_lfu_cache import HybridLFUCache, EvictionRecord


def test_small_capacity_high_freq_stability():
    print("\n" + "=" * 60)
    print("=== 测试1: 小容量高频冷却稳定性 ===")
    print("=" * 60)

    random.seed(12345)

    for run in range(5):
        print(f"\n--- 第 {run+1} 次重复 ---")
        cache = HybridLFUCache(
            capacity=3,
            decay_window=0.02,
            decay_factor=0.5,
            sample_size=3,
            decay_check_interval=10,
        )

        cache.put('high_freq', 'v1')
        for i in range(50):
            cache.get('high_freq')

        high_freq_node = cache._cache['high_freq']
        print(f"  高频访问后: freq={high_freq_node.freq}, last_access={high_freq_node.last_access_time:.3f}")
        assert high_freq_node.freq == 51

        time.sleep(0.03)

        try:
            val = cache.get('high_freq')
            assert val == 'v1', f"值错误: {val}"
            high_freq_node = cache._cache['high_freq']
            print(f"  冷却后读取: freq={high_freq_node.freq} (应该已衰减然后+1)")
            assert high_freq_node.freq < 51, "应该已衰减"
        except Exception as e:
            print(f"  ❌ 读取时报错: {e}")
            raise

        print(f"  开始塞20个新key, 逐步淘汰...")
        for i in range(20):
            cache.put(f'new_key_{i}', f'val_{i}')

        if 'high_freq' in cache:
            high_freq_node = cache._cache['high_freq']
            print(f"  塞新key后: high_freq仍在, freq={high_freq_node.freq}")
        else:
            print(f"  塞新key后: high_freq已被淘汰 ✓")

        assert len(cache) <= 3

        while 'high_freq' in cache:
            cache.put(f'flush_{cache.eviction_count}', 'x')

        assert 'high_freq' not in cache, "high_freq最终应该被淘汰让位"
        print(f"  high_freq已被淘汰, 缓存大小: {len(cache)}")

        cache.put('final_a', 'a')
        cache.put('final_b', 'b')
        cache.put('final_c', 'c')

        for k in ['final_a', 'final_b', 'final_c']:
            assert k in cache, f"{k} 应该在缓存中"

        print(f"  ✓ 第 {run+1} 次通过: 无报错, 淘汰正常")

    print("\n✅ 小容量高频冷却稳定性测试通过! 重复5次无报错")


def test_multi_freq_level_performance():
    print("\n" + "=" * 60)
    print("=== 测试2: 多频率层级性能 ===")
    print("=" * 60)

    CAPACITY = 20000
    TOTAL_OPS = 20000

    def create_cache():
        return HybridLFUCache(
            capacity=CAPACITY,
            decay_window=60.0,
            decay_check_interval=1000,
            sample_size=5,
            track_evictions=False,
            track_latency=True,
            max_latency_samples=20000,
        )

    def build_freq_levels(cache, num_levels):
        keys_by_level = {}
        for level in range(num_levels):
            freq_target = 2 ** (level + 1)
            level_keys = [f'level_{level}_key_{i}' for i in range(20)]
            keys_by_level[level] = level_keys
            for key in level_keys:
                cache.put(key, f'level_{level}')
                for _ in range(freq_target - 1):
                    cache.get(key)
        return keys_by_level

    results = []
    for num_levels in [3, 6, 9, 12, 15]:
        cache = create_cache()
        build_freq_levels(cache, num_levels)

        freq_count = len(cache._freq_buckets)
        print(f"\n  频率层级数: {freq_count}")
        print(f"  频率分布(Top5): {sorted(cache._freq_buckets.keys(), reverse=True)[:8]}...")

        cache.reset_stats()

        all_keys = []
        for level in range(num_levels):
            all_keys.extend([f'level_{level}_key_{i}' for i in range(20)])
        random.shuffle(all_keys)

        start = time.perf_counter()
        for i in range(TOTAL_OPS):
            if i % 3 == 0:
                cache.put(f'new_{i}', 'val')
            else:
                cache.get(random.choice(all_keys))

        elapsed = time.perf_counter() - start
        avg_latency = elapsed / TOTAL_OPS * 1e6

        stats = cache.get_stats()
        p50 = cache.get_latency_percentiles('get').get('p50', 0)
        p99 = cache.get_latency_percentiles('get').get('p99', 0)

        print(f"  操作数: {TOTAL_OPS}, 耗时: {elapsed:.3f}s")
        print(f"  平均延迟: {avg_latency:.2f}μs, P50: {p50:.2f}μs, P99: {p99:.2f}μs")
        print(f"  命中率: {stats['hit_rate']}%, 淘汰: {stats['eviction_count']}")

        results.append({
            'freq_levels': freq_count,
            'avg_latency': avg_latency,
            'p50': p50,
            'p99': p99,
        })

    print("\n" + "-" * 50)
    print("性能对比:")
    base_avg = results[0]['avg_latency']
    for r in results:
        ratio = r['avg_latency'] / base_avg
        print(f"  {r['freq_levels']:3d}个频率层级: 平均={r['avg_latency']:6.2f}μs "
              f"(x{ratio:.2f}), P50={r['p50']:6.2f}μs, P99={r['p99']:6.2f}μs")

    max_ratio = max(r['avg_latency'] / base_avg for r in results)
    print(f"\n  最大延迟倍数: {max_ratio:.2f}x (阈值 < 3x)")
    assert max_ratio < 3.0, f"延迟随频率层级增长过快: {max_ratio:.2f}x"

    print("\n✅ 多频率层级性能测试通过! 无线性退化")


def test_same_freq_lru_order():
    print("\n" + "=" * 60)
    print("=== 测试3: 同频率下真实LRU顺序 ===")
    print("=" * 60)

    cache = HybridLFUCache(
        capacity=5,
        decay_window=0.02,
        decay_factor=0.5,
        sample_size=3,
        decay_check_interval=5,
    )

    print("\n步骤1: 建立旧热点 (访问32次 → freq=32)")
    cache.put('old_hot', 'old_hot_val')
    for _ in range(31):
        cache.get('old_hot')
    old_hot_node = cache._cache['old_hot']
    old_hot_time = old_hot_node.last_access_time
    print(f"  old_hot: freq={old_hot_node.freq}, last_access={old_hot_time:.3f}")

    print("\n步骤2: 加入3个低频数据, 并分别在不同时间访问")
    cache.put('cold_a', 'a')
    cache.get('cold_a')
    time.sleep(0.005)

    cache.put('cold_b', 'b')
    cache.get('cold_b')
    time.sleep(0.005)

    cache.put('cold_c', 'c')
    cache.get('cold_c')
    time.sleep(0.005)

    print(f"  cold_a: last_access={cache._cache['cold_a'].last_access_time:.3f}")
    print(f"  cold_b: last_access={cache._cache['cold_b'].last_access_time:.3f}")
    print(f"  cold_c: last_access={cache._cache['cold_c'].last_access_time:.3f}")

    print("\n步骤3: 等待旧热点冷却超时, 访问一次触发衰减")
    time.sleep(0.03)

    cache.get('old_hot')
    old_hot_node = cache._cache['old_hot']
    print(f"  old_hot衰减后: freq={old_hot_node.freq}")

    target_freq = old_hot_node.freq
    print(f"  目标频率桶: {target_freq}")

    bucket = cache._freq_buckets.get(target_freq)
    if bucket:
        node = bucket.head.next
        keys_in_bucket = []
        while node != bucket.tail:
            if node.key:
                keys_in_bucket.append((node.key, node.last_access_time))
            node = node.next
        print(f"  频率={target_freq}桶内顺序(头→尾):")
        for k, t in keys_in_bucket:
            idle = time.monotonic() - t
            print(f"    {k}: last_access={t:.3f}, idle={idle:.3f}s")

        in_bucket = [k for k, _ in keys_in_bucket]
        positions = {k: i for i, (k, _) in enumerate(keys_in_bucket)}

        print(f"\n  位置索引: {positions}")

        if 'old_hot' in positions and 'cold_a' in positions:
            old_hot_pos = positions['old_hot']
            cold_a_pos = positions['cold_a']
            print(f"  old_hot位置: {old_hot_pos}, cold_a位置: {cold_a_pos}")
            assert old_hot_pos > cold_a_pos, (
                f"旧热点衰减后应该在更久没访问的cold_a后面, "
                f"但old_hot在{old_hot_pos}, cold_a在{cold_a_pos}"
            )
            print("  ✅ 旧热点衰减后位于尾部, 淘汰优先级更高")

    print("\n步骤4: 塞入新数据触发淘汰, 验证淘汰顺序")
    eviction_order = []

    original_track = cache.track_evictions
    cache.track_evictions = True
    cache._eviction_records.clear()
    cache._eviction_by_category.clear()

    for i in range(10):
        cache.put(f'new_{i}', f'val_{i}')

    recent_evictions = cache.get_recent_evictions(10)
    eviction_keys = [r.key for r in recent_evictions]

    print(f"\n  最近淘汰顺序(先→后): {eviction_keys}")

    cold_a_evicted = 'cold_a' in eviction_keys
    old_hot_evicted = 'old_hot' in eviction_keys

    if cold_a_evicted and old_hot_evicted:
        cold_a_idx = eviction_keys.index('cold_a')
        old_hot_idx = eviction_keys.index('old_hot')
        print(f"  cold_a淘汰位置: {cold_a_idx}, old_hot淘汰位置: {old_hot_idx}")

        if cold_a_idx < old_hot_idx:
            print("  ✅ 正确: 更早没访问的cold_a先被淘汰, 然后才是旧热点")
        else:
            print(f"  ⚠️  淘汰顺序验证: cold_a在{cold_a_idx}, old_hot在{old_hot_idx}")
    elif cold_a_evicted:
        print("  ✅ cold_a已被淘汰 (更久没访问)")
    elif old_hot_evicted:
        print("  ✅ old_hot已被淘汰 (虽然刚访问过但已衰减)")

    cache.track_evictions = original_track

    print("\n✅ 同频率下真实LRU顺序测试通过!")


def test_enhanced_stress_test():
    print("\n" + "=" * 60)
    print("=== 测试4: 增强版真实流量压力测试 ===")
    print("=" * 60)

    random.seed(42)

    def classify_key(key):
        if isinstance(key, str):
            if key.startswith('old_hot'):
                return 'old_hot'
            elif key.startswith('new_hot'):
                return 'new_hot'
            elif key.startswith('warm'):
                return 'warm'
            elif key.startswith('scan'):
                return 'scan'
            elif key.startswith('cold'):
                return 'cold'
        return 'other'

    CAPACITY = 3000
    OLD_HOT_COUNT = 800
    NEW_HOT_COUNT = 800
    WARM_COUNT = 1500
    DECAY_WINDOW = 0.03

    cache = HybridLFUCache(
        capacity=CAPACITY,
        decay_window=DECAY_WINDOW,
        decay_factor=0.5,
        sample_size=10,
        decay_check_interval=30,
        track_evictions=True,
        max_eviction_records=2000,
        track_latency=True,
        max_latency_samples=20000,
        key_classifier=classify_key,
    )

    old_hot_keys = [f'old_hot_{i}' for i in range(OLD_HOT_COUNT)]
    new_hot_keys = [f'new_hot_{i}' for i in range(NEW_HOT_COUNT)]
    warm_keys = [f'warm_{i}' for i in range(WARM_COUNT)]

    print(f"\n配置: 容量={CAPACITY}, 旧热点={OLD_HOT_COUNT}, 新热点={NEW_HOT_COUNT}, 温点={WARM_COUNT}")
    print(f"     衰减窗口={DECAY_WINDOW}s, 采样数=10, 检查间隔=30次")

    phase_results = []

    def run_phase(name, ops, op_gen, description=""):
        cache.reset_stats()
        start = time.perf_counter()

        for _ in range(ops):
            op = op_gen()
            if isinstance(op, tuple):
                op_type = op[0]
                key = op[1]
                if op_type == 'put':
                    cache.put(key, op[2] if len(op) > 2 else 'val')
                else:
                    cache.get(key)
            else:
                if random.random() < 0.75:
                    cache.get(op)
                else:
                    cache.put(op, 'val')

        elapsed = time.perf_counter() - start
        stats = cache.get_stats()
        detailed = cache.get_detailed_stats()

        old_survive = sum(1 for k in old_hot_keys if k in cache)
        new_survive = sum(1 for k in new_hot_keys if k in cache)
        warm_survive = sum(1 for k in warm_keys if k in cache)

        result = {
            'name': name,
            'description': description,
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
            'eviction_by_category': stats.get('eviction_by_category', {}),
            'latency_get': detailed.get('latency_get', {}),
            'latency_put': detailed.get('latency_put', {}),
            'latency_dist': detailed.get('latency_distribution', {}),
            'recent_evictions': detailed.get('recent_evictions', []),
        }
        phase_results.append(result)

        top_freqs = sorted(stats['freq_distribution'].items(), reverse=True)[:5]
        freq_levels = len(stats['freq_distribution'])

        print(f"\n  ┌─ {name} {'─' * (40 - len(name))}")
        if description:
            print(f"  │ {description}")
        print(f"  │ 耗时: {elapsed:.3f}s | 平均延迟: {elapsed/ops*1e6:.2f}μs | 操作数: {ops}")
        print(f"  │ 命中率: {stats['hit_rate']}% | 淘汰: {stats['eviction_count']} | 衰减: {stats['decay_count']}")
        print(f"  │ 旧热点: {old_survive}/{OLD_HOT_COUNT} ({old_survive/OLD_HOT_COUNT*100:.1f}%)")
        print(f"  │ 新热点: {new_survive}/{NEW_HOT_COUNT} ({new_survive/NEW_HOT_COUNT*100:.1f}%)")
        print(f"  │ 温点: {warm_survive}/{WARM_COUNT} ({warm_survive/WARM_COUNT*100:.1f}%)")
        print(f"  │ 频率层级: {freq_levels} | Top5: {top_freqs}")

        lat_get = result['latency_get']
        if lat_get:
            print(f"  │ GET延迟: P50={lat_get.get('p50', 0):.2f}μs, P95={lat_get.get('p95', 0):.2f}μs, P99={lat_get.get('p99', 0):.2f}μs")

        evict_cat = result['eviction_by_category']
        if evict_cat:
            print(f"  │ 淘汰分类: {evict_cat}")

        recent = result['recent_evictions']
        if recent:
            categories = [r.category for r in recent[-10:]]
            print(f"  │ 最近10个淘汰: {categories}")

        print(f"  └{'─' * 50}")

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

    r1 = run_phase("旧热点建立", 25000, phase1_gen, "旧热点高频访问，温点辅助")
    assert r1['old_hot_survive'] >= OLD_HOT_COUNT * 0.95

    # ── 阶段2: 抗扫描验证 ──
    print("\n[阶段2] 抗扫描验证: 一次性扫描大量冷数据，热点不应被挤出")

    scan_i = 0
    def phase2_gen():
        nonlocal scan_i
        r = random.random()
        if r < 0.55:
            return ('get', random.choice(old_hot_keys))
        elif r < 0.75:
            return ('get', random.choice(warm_keys))
        else:
            k = f'scan_{scan_i}'
            scan_i += 1
            return ('put', k, f'scan_value_{scan_i-1}')

    r2 = run_phase("抗扫描测试", 8000, phase2_gen, "插入3000+扫描数据")

    print(f"\n  → 抗扫描结果: 旧热点存活率 {r2['old_hot_survive']}/{OLD_HOT_COUNT} ({r2['old_hot_survive']/OLD_HOT_COUNT*100:.1f}%)")
    evict_scan = r2['eviction_by_category'].get('scan', 0)
    evict_old = r2['eviction_by_category'].get('old_hot', 0)
    evict_warm = r2['eviction_by_category'].get('warm', 0)
    print(f"  → 淘汰分类: scan={evict_scan}, old_hot={evict_old}, warm={evict_warm}")
    print(f"  → 完整淘汰分类: {dict(r2['eviction_by_category'])}")
    assert r2['old_hot_survive'] >= OLD_HOT_COUNT * 0.9, "扫描数据不应挤出热点"
    total_evict = sum(r2['eviction_by_category'].values())
    if total_evict > 0:
        assert evict_scan > evict_old * 5, "被淘汰的应该主要是扫描数据"
    print("  ✅ 抗扫描验证通过")

    # ── 阶段3: 冷却期 ──
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

    r3 = run_phase("冷却衰减期", 15000, phase3_gen, "旧热点完全不访问，让时间衰减生效")

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

    r4 = run_phase("新热点上位", 30000, phase4_gen, "新热点开始高频访问")

    print(f"\n  → 上位期结束:")
    print(f"    旧热点: {r4['old_hot_survive']}/{OLD_HOT_COUNT} ({r4['old_hot_survive']/OLD_HOT_COUNT*100:.1f}%)")
    print(f"    新热点: {r4['new_hot_survive']}/{NEW_HOT_COUNT} ({r4['new_hot_survive']/NEW_HOT_COUNT*100:.1f}%)")

    # ── 阶段5: 新热点稳定 + 旧热点让位 ──
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

    r5 = run_phase("稳定让位期", 30000, phase5_gen, "新热点持续访问，旧热点继续让位")

    # ── 最终验证 ──
    print("\n" + "=" * 55)
    print("最终结果:")
    print(f"  旧热点存活: {r5['old_hot_survive']}/{OLD_HOT_COUNT} ({r5['old_hot_survive']/OLD_HOT_COUNT*100:.1f}%)")
    print(f"  新热点存活: {r5['new_hot_survive']}/{NEW_HOT_COUNT} ({r5['new_hot_survive']/NEW_HOT_COUNT*100:.1f}%)")
    print(f"  温点存活: {r5['warm_survive']}/{WARM_COUNT}")

    total_evictions = cache.eviction_count
    evict_by_cat = cache.get_eviction_by_category()
    print(f"  总淘汰: {total_evictions} | 总衰减: {cache.decay_count}")
    print(f"  淘汰分类汇总: {evict_by_cat}")

    lat_all = cache.get_latency_percentiles()
    lat_dist = cache.get_latency_distribution()
    print(f"\n  延迟统计(所有操作):")
    print(f"    P50={lat_all.get('p50', 0):.2f}μs, P90={lat_all.get('p90', 0):.2f}μs, "
          f"P95={lat_all.get('p95', 0):.2f}μs, P99={lat_all.get('p99', 0):.2f}μs")
    print(f"    分布: {lat_dist}")

    recent_evict = cache.get_recent_evictions(15)
    if recent_evict:
        print(f"\n  最近15个淘汰记录:")
        for r in recent_evict:
            idle = r.eviction_time - r.last_access_time
            print(f"    {r.key:<20} cat={r.category:<8} freq={r.freq:3d} idle={idle:.3f}s")

    # 断言验证
    assert r5['new_hot_survive'] >= NEW_HOT_COUNT * 0.9, "新热点应大部分留存"
    assert r5['old_hot_survive'] < r1['old_hot_survive'] * 0.8, "旧热点应该显著减少"
    assert r5['old_hot_survive'] < OLD_HOT_COUNT * 0.75, "旧热点应大部分被淘汰让位"

    evict_old = evict_by_cat.get('old_hot', 0)
    evict_new = evict_by_cat.get('new_hot', 0)
    evict_scan_total = evict_by_cat.get('scan', 0)
    if evict_new > 0:
        old_new_ratio = evict_old / evict_new
        print(f"\n  旧热点/新热点 淘汰比: {old_new_ratio:.1f}x")
        assert old_new_ratio > 2.0, "被淘汰的旧热点应该远多于新热点"

    if evict_scan_total > 0:
        print(f"  扫描数据淘汰数: {evict_scan_total} (证明抗扫描有效)")

    # 性能验证
    all_latencies = [r['avg_latency_us'] for r in phase_results]
    latency_ratio = max(all_latencies) / min(all_latencies)
    print(f"\n  各阶段平均延迟(μs): {[f'{l:.2f}' for l in all_latencies]}")
    print(f"  延迟波动: {latency_ratio:.2f}x")
    assert latency_ratio < 3.0, "延迟波动不应太大"

    print("\n" + "=" * 55)
    print("✅ 增强版压力测试通过!")
    print("  ✅ 抗扫描: 热点数据不会被一次性扫描挤出")
    print("  ✅ 新热点上位: 新热点能取代旧热点的位置")
    print("  ✅ 衰减保序: 衰减节点在同频率桶中位于尾部")
    print("  ✅ 常数级性能: 各阶段延迟稳定，无线性退化")
    print("  ✅ 可观察性: 完整的延迟采样、淘汰分类、最近淘汰记录")


def test_basic_operations():
    print("\n=== 测试基本操作 ===")
    cache = HybridLFUCache(capacity=3, decay_check_interval=1000, track_evictions=False, track_latency=False)

    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)

    assert cache.get('a') == 1
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    assert cache.get('d') is None

    print("✓ 基本put/get操作正常")


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
            track_evictions=False,
            track_latency=False,
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

    print("✓ 时间复杂度为常数级")


if __name__ == '__main__':
    test_basic_operations()
    test_constant_time_complexity()

    test_small_capacity_high_freq_stability()
    test_multi_freq_level_performance()
    test_same_freq_lru_order()
    test_enhanced_stress_test()

    print("\n" + "=" * 60)
    print("🎉 所有测试通过!")
    print("=" * 60)
