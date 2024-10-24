import cityhash
from array import array
import random
import math
import time
import sys
import tabulate
from typing import Dict, List, Any
import matplotlib.pyplot as plt

class FarmCuckooFilter:
    def __init__(self, capacity: int = 3_000_000, fpr: float = 0.0005):
        # Use same sizing as XOR for fairness
        self.capacity = capacity
        self.size = math.ceil(1.23 * capacity)
        self.fingerprint_size = math.ceil(math.log2(1/fpr))
        # Use single flat array like XOR
        self.fingerprints = array('H', [0] * self.size)
        
    def _get_hash(self, item: bytes, seed: bytes) -> int:
        """Same hash function as XOR"""
        if isinstance(item, str):
            item = item.encode()
        return cityhash.CityHash64(item + seed) % self.size
    
    def insert(self, item) -> bool:
        """Simplified insert with same operations as XOR"""
        if isinstance(item, str):
            item = item.encode()
        
        # Use exact same hash function calls as XOR
        h1 = self._get_hash(item, b'1')
        h2 = self._get_hash(item, b'2')
        # Third hash used for alternate location instead of XOR's third slot
        h3 = self._get_hash(item, b'3')
        
        # Try first pair of locations
        if self.fingerprints[h1] == 0:
            self.fingerprints[h1] = 1
            return True
        if self.fingerprints[h2] == 0:
            self.fingerprints[h2] = 1
            return True
        
        # Use h3 as alternate location if first two are full
        if self.fingerprints[h3] == 0:
            self.fingerprints[h3] = 1
            return True
            
        return False
    
    def contains(self, item) -> bool:
        if isinstance(item, str):
            item = item.encode()
        
        h1 = self._get_hash(item, b'1')
        h2 = self._get_hash(item, b'2')
        h3 = self._get_hash(item, b'3')
        
        # Check any of the possible locations
        return bool(self.fingerprints[h1] or self.fingerprints[h2] or self.fingerprints[h3])

    def get_stats(self) -> dict:
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fingerprint_size': self.fingerprint_size,
            'memory_bytes': sys.getsizeof(self.fingerprints)
        }

class FarmXORFilter:
    def __init__(self, capacity: int, fpr: float = 0.0005):
        self.capacity = capacity
        self.size = math.ceil(1.23 * capacity)
        self.fingerprint_size = math.ceil(math.log2(1/fpr))
        self.fingerprints = array('H', [0] * self.size)
        
    def _get_hash(self, item: bytes, seed: bytes) -> int:
        if isinstance(item, str):
            item = item.encode()
        return cityhash.CityHash64(item + seed) % self.size
        
    def insert(self, item) -> bool:
        if isinstance(item, str):
            item = item.encode()
        
        h1 = self._get_hash(item, b'1')
        h2 = self._get_hash(item, b'2')
        h3 = self._get_hash(item, b'3')
        
        # Mark all three locations
        self.fingerprints[h1] = 1
        self.fingerprints[h2] = 1
        self.fingerprints[h3] = 1
        
        return True
        
    def contains(self, item) -> bool:
        if isinstance(item, str):
            item = item.encode()
        
        h1 = self._get_hash(item, b'1')
        h2 = self._get_hash(item, b'2')
        h3 = self._get_hash(item, b'3')
        
        return bool(self.fingerprints[h1] and self.fingerprints[h2] and self.fingerprints[h3])

    def get_stats(self) -> dict:
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fingerprint_size': self.fingerprint_size,
            'memory_bytes': sys.getsizeof(self.fingerprints)
        }

def run_comparison_benchmarks(sizes: List[int] = [100_000, 1_000_000, 3_000_000], 
                            test_lookups: int = 100_000,
                            sample_size: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
    results = {
        'cuckoo': [],
        'xor': []
    }
    
    for num_items in sizes:
        print(f"\nBenchmarking with {num_items:,} items...")
        
        print("Generating test data...")
        items = [f"item{i}".encode() for i in range(num_items)]
        test_items = [f"item{i}".encode() for i in range(num_items, num_items + test_lookups)]
        
        # Select sample indices for detailed timing
        sample_indices = random.sample(range(num_items), min(sample_size, num_items))
        lookup_sample_indices = random.sample(range(test_lookups), min(sample_size, test_lookups))
        
        for filter_type in ['cuckoo', 'xor']:
            print(f"\nTesting {filter_type.upper()} filter...")
            
            start_time = time.time()
            if filter_type == 'cuckoo':
                f = FarmCuckooFilter(num_items)
            else:
                f = FarmXORFilter(num_items)
            init_time = time.time() - start_time
            
            insert_times = []
            successful = 0
            start_time = time.time()
            
            for i, item in enumerate(items):
                if i in sample_indices:
                    item_start = time.time()
                    success = f.insert(item)
                    insert_times.append(time.time() - item_start)
                else:
                    success = f.insert(item)
                
                if success:
                    successful += 1
            
            total_insert_time = time.time() - start_time
            
            print("Testing lookups...")
            lookup_times = []
            start_time = time.time()
            
            for i, item in enumerate(test_items):
                if i in lookup_sample_indices:
                    item_start = time.time()
                    f.contains(item)
                    lookup_times.append(time.time() - item_start)
                else:
                    f.contains(item)
            
            total_lookup_time = time.time() - start_time
            
            memory_bytes = sys.getsizeof(f.fingerprints)
            memory_mb = memory_bytes / (1024 * 1024)
            
            insert_latencies = sorted([t * 1000000 for t in insert_times])
            lookup_latencies = sorted([t * 1000000 for t in lookup_times])
            
            def percentile(lst, p):
                if not lst:
                    return 0
                return lst[int(len(lst) * p)]
            
            results[filter_type].append({
                'num_items': num_items,
                'memory_mb': memory_mb,
                'init_time': init_time,
                'total_insert_time': total_insert_time,
                'avg_insert_time_us': (total_insert_time/num_items)*1000000,
                'insert_throughput': num_items/total_insert_time,
                'total_lookup_time': total_lookup_time,
                'avg_lookup_time_us': (total_lookup_time/test_lookups)*1000000,
                'lookup_throughput': test_lookups/total_lookup_time,
                'insert_p50': percentile(insert_latencies, 0.5),
                'insert_p95': percentile(insert_latencies, 0.95),
                'insert_p99': percentile(insert_latencies, 0.99),
                'lookup_p50': percentile(lookup_latencies, 0.5),
                'lookup_p95': percentile(lookup_latencies, 0.95),
                'lookup_p99': percentile(lookup_latencies, 0.99),
                'successful_inserts': successful
            })
            
            del f
            
    return results

def print_comparison_table(results: Dict[str, List[Dict[str, Any]]]):
    headers = ['Metric', 'Filter Type', '100K items', '1M items', '3M items']
    table_data = []
    
    metrics = [
        ('Memory (MB)', 'memory_mb', '.2f'),
        ('Init Time (s)', 'init_time', '.4f'),
        ('Insert Time (s)', 'total_insert_time', '.2f'),
        ('Avg Insert (μs)', 'avg_insert_time_us', '.2f'),
        ('Insert Rate (items/s)', 'insert_throughput', ',.0f'),
        ('Lookup Time (s)', 'total_lookup_time', '.2f'),
        ('Avg Lookup (μs)', 'avg_lookup_time_us', '.2f'),
        ('Lookup Rate (ops/s)', 'lookup_throughput', ',.0f'),
        ('Insert p99 (μs)', 'insert_p99', '.2f'),
        ('Lookup p99 (μs)', 'lookup_p99', '.2f')
    ]
    
    for metric_name, metric_key, fmt in metrics:
        for filter_type in ['Cuckoo', 'XOR']:
            row = [metric_name, filter_type]
            for i in range(3):
                value = results[filter_type.lower()][i][metric_key]
                row.append(f'{value:{fmt}}')
            table_data.append(row)
        table_data.append([''] * 5)
    
    print(tabulate.tabulate(table_data, headers=headers, tablefmt='grid'))

def plot_comparison(results: Dict[str, List[Dict[str, Any]]]):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    sizes = [r['num_items'] for r in results['cuckoo']]
    
    # Memory usage
    ax1.plot(sizes, [r['memory_mb'] for r in results['cuckoo']], 'b-', label='Cuckoo')
    ax1.plot(sizes, [r['memory_mb'] for r in results['xor']], 'r--', label='XOR')
    ax1.set_title('Memory Usage')
    ax1.set_xlabel('Number of Items')
    ax1.set_ylabel('Memory (MB)')
    ax1.legend()
    
    # Insert throughput
    ax2.plot(sizes, [r['insert_throughput'] for r in results['cuckoo']], 'b-', label='Cuckoo')
    ax2.plot(sizes, [r['insert_throughput'] for r in results['xor']], 'r--', label='XOR')
    ax2.set_title('Insert Throughput')
    ax2.set_xlabel('Number of Items')
    ax2.set_ylabel('Items/second')
    ax2.legend()
    
    # Lookup throughput
    ax3.plot(sizes, [r['lookup_throughput'] for r in results['cuckoo']], 'b-', label='Cuckoo')
    ax3.plot(sizes, [r['lookup_throughput'] for r in results['xor']], 'r--', label='XOR')
    ax3.set_title('Lookup Throughput')
    ax3.set_xlabel('Number of Items')
    ax3.set_ylabel('Lookups/second')
    ax3.legend()
    
    # P99 latencies
    ax4.plot(sizes, [r['insert_p99'] for r in results['cuckoo']], 'b-', label='Cuckoo Insert')
    ax4.plot(sizes, [r['insert_p99'] for r in results['xor']], 'r--', label='XOR Insert')
    ax4.plot(sizes, [r['lookup_p99'] for r in results['cuckoo']], 'g-', label='Cuckoo Lookup')
    ax4.plot(sizes, [r['lookup_p99'] for r in results['xor']], 'm--', label='XOR Lookup')
    ax4.set_title('P99 Latencies')
    ax4.set_xlabel('Number of Items')
    ax4.set_ylabel('Microseconds')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png')
    print("\nComparison plots saved as 'filter_comparison.png'")

if __name__ == "__main__":
    print("Running filter comparison benchmarks...")
    results = run_comparison_benchmarks()
    print("\nDetailed Comparison Results:")
    print_comparison_table(results)
    plot_comparison(results)