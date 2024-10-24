import mmh3
from array import array
import random
import math
import time
import sys
import tabulate
from typing import Dict, List, Any
import matplotlib.pyplot as plt
class MurmurCuckooFilter:

    def __init__(self, capacity: int = 3_000_000, fpr: float = 0.0005):
        """
        Initialize Cuckoo Filter using Murmur3 hashing
        capacity: Expected number of items (e.g., 3 million)
        fpr: False positive rate (e.g., 0.0005 for 0.05%)
        """
        self.bucket_size = 4
        self.fingerprint_size = math.ceil(math.log2(1/fpr))  # ~12 bits for 0.05% FPR
        self.num_buckets = math.ceil(capacity / self.bucket_size * 1.05)
        if self.num_buckets % 2 == 0:
            self.num_buckets += 1
            
        self.buckets = [array('H', [0] * self.bucket_size) for _ in range(self.num_buckets)]
        self.fingerprint_mask = (1 << self.fingerprint_size) - 1
        self.size = 0
        
        self.SEED_FINGERPRINT = 42
        self.SEED_INDEX1 = 123
        self.SEED_INDEX2 = 321
    
    def _get_fingerprint(self, item: bytes) -> int:
        """
        Generate fingerprint using Murmur3 hash
        Returns non-zero fingerprint of specified size
        """
        if isinstance(item, str):
            item = item.encode()
            
        fingerprint = mmh3.hash(item, seed=self.SEED_FINGERPRINT) & self.fingerprint_mask
        return max(1, fingerprint) 
    
    def _get_indices(self, item: bytes, fingerprint: int) -> tuple[int, int]:
        """
        Calculate both possible bucket locations using Murmur3
        Returns tuple of (index1, index2)
        """
        if isinstance(item, str):
            item = item.encode()
            
        index1 = mmh3.hash(item, seed=self.SEED_INDEX1) % self.num_buckets
        
        hash2 = mmh3.hash(str(fingerprint).encode(), seed=self.SEED_INDEX2)
        index2 = (index1 ^ hash2) % self.num_buckets
        
        return index1, index2
    
    def insert(self, item) -> bool:
        """Insert item into the filter"""
        if isinstance(item, str):
            item = item.encode()
            
        fingerprint = self._get_fingerprint(item)
        
        index1, index2 = self._get_indices(item, fingerprint)
        
        for idx in [index1, index2]:
            for pos in range(self.bucket_size):
                if self.buckets[idx][pos] == 0:  # Empty slot
                    self.buckets[idx][pos] = fingerprint
                    self.size += 1
                    return True
        
        current_idx = random.choice([index1, index2])
        current_fingerprint = fingerprint
        
        for _ in range(500):
            kick_pos = random.randrange(self.bucket_size)
            
            current_fingerprint, self.buckets[current_idx][kick_pos] = \
                self.buckets[current_idx][kick_pos], current_fingerprint
            
            hash2 = mmh3.hash(str(current_fingerprint).encode(), seed=self.SEED_INDEX2)
            current_idx = (current_idx ^ hash2) % self.num_buckets
            
            for pos in range(self.bucket_size):
                if self.buckets[current_idx][pos] == 0:
                    self.buckets[current_idx][pos] = current_fingerprint
                    self.size += 1
                    return True
        
        return False 
    
    def contains(self, item) -> bool:
        """Check if item might be in filter"""
        if isinstance(item, str):
            item = item.encode()
            
        fingerprint = self._get_fingerprint(item)
        index1, index2 = self._get_indices(item, fingerprint)
        
        return (fingerprint in self.buckets[index1] or 
                fingerprint in self.buckets[index2])

    def get_stats(self) -> dict:
        """Get filter statistics"""
        return {
            'size': self.size,
            'capacity': self.num_buckets * self.bucket_size,
            'load_factor': self.size / (self.num_buckets * self.bucket_size),
            'num_buckets': self.num_buckets,
            'fingerprint_size': self.fingerprint_size,
            'bucket_size': self.bucket_size
        }

def demo_murmur_cuckoo():
    cf = MurmurCuckooFilter()
    
    items = [f"item{i}".encode() for i in range(10)]
    for item in items:
        cf.insert(item)
    
    print("Filter stats:", cf.get_stats())
    print("\nLookup examples:")
    print("'item1' in filter:", cf.contains("item1"))
    print("'item99' in filter:", cf.contains("item99"))
    
    test_item = "test_item".encode()
    fp = cf._get_fingerprint(test_item)
    idx1, idx2 = cf._get_indices(test_item, fp)
    print(f"\nFor item 'test_item':")
    print(f"Fingerprint: {fp:x}")
    print(f"Bucket indices: {idx1}, {idx2}")

class XORFilter:
    def __init__(self, capacity: int, fpr: float = 0.0005):
        self.capacity = capacity
        self.size = math.ceil(1.23 * capacity)
        self.fingerprint_size = math.ceil(math.log2(1/fpr))
        self.fingerprints = array('H', [0] * self.size)
        self.size_mask = self.size - 1
        
        self.SEED1 = 42
        self.SEED2 = 123
        self.SEED3 = 321
        
    def _get_hash(self, item: bytes, seed: int) -> int:
        """Get hash location"""
        if isinstance(item, str):
            item = item.encode()
        return mmh3.hash(item, seed) % self.size
        
    def insert(self, item) -> bool:
        """Insert item into filter"""
        if isinstance(item, str):
            item = item.encode()
            
        h1 = self._get_hash(item, self.SEED1)
        h2 = self._get_hash(item, self.SEED2)
        h3 = self._get_hash(item, self.SEED3)
        
        fingerprint = mmh3.hash(item, seed=0) & ((1 << self.fingerprint_size) - 1)
        
        self.fingerprints[h1] ^= fingerprint
        self.fingerprints[h2] ^= fingerprint
        self.fingerprints[h3] ^= fingerprint
        
        return True
        
    def contains(self, item) -> bool:
        """Check if item might be in filter"""
        if isinstance(item, str):
            item = item.encode()
            
        h1 = self._get_hash(item, self.SEED1)
        h2 = self._get_hash(item, self.SEED2)
        h3 = self._get_hash(item, self.SEED3)
        
        fingerprint = mmh3.hash(item, seed=0) & ((1 << self.fingerprint_size) - 1)
        
        return fingerprint == (self.fingerprints[h1] ^ self.fingerprints[h2] ^ self.fingerprints[h3])
        
    def get_stats(self) -> dict:
        """Get filter statistics"""
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fingerprint_size': self.fingerprint_size,
            'memory_bytes': sys.getsizeof(self.fingerprints)
        }

def run_comparison_benchmarks(sizes: List[int] = [100_000, 1_000_000, 3_000_000], 
                            test_lookups: int = 100_000) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run comprehensive benchmarks comparing Cuckoo and XOR filters
    """
    results = {
        'cuckoo': [],
        'xor': []
    }
    
    for num_items in sizes:
        print(f"\nBenchmarking with {num_items:,} items...")
        
        # Test data generation
        print("Generating test data...")
        items = [f"item{i}".encode() for i in range(num_items)]
        test_items = [f"item{i}".encode() for i in range(num_items, num_items + test_lookups)]
        
        # Test both filter types
        for filter_type in ['cuckoo', 'xor']:
            print(f"\nTesting {filter_type.upper()} filter...")
            
            # Initialize filter
            start_time = time.time()
            if filter_type == 'cuckoo':
                f = MurmurCuckooFilter(num_items)
            else:
                f = XORFilter(num_items)
            init_time = time.time() - start_time
            
            # Measure insertion
            insert_times = []
            successful = 0
            start_time = time.time()
            
            for item in items:
                item_start = time.time()
                if f.insert(item):
                    successful += 1
                insert_times.append(time.time() - item_start)
            
            total_insert_time = time.time() - start_time
            
            lookup_times = []
            start_time = time.time()
            
            for item in test_items:
                item_start = time.time()
                f.contains(item)
                lookup_times.append(time.time() - item_start)
            
            total_lookup_time = time.time() - start_time
            
            if filter_type == 'cuckoo':
                memory_bytes = sum(sys.getsizeof(bucket) for bucket in f.buckets)
            else:
                memory_bytes = sys.getsizeof(f.fingerprints)
            
            memory_mb = memory_bytes / (1024 * 1024)
            
            insert_latencies = sorted([t * 1000000 for t in insert_times])
            lookup_latencies = sorted([t * 1000000 for t in lookup_times])
            
            def percentile(lst, p):
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
                'lookup_p99': percentile(lookup_latencies, 0.99)
            })
            
    return results

def print_comparison_table(results: Dict[str, List[Dict[str, Any]]]):
    """Print a formatted comparison table"""
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
    """Create comparison plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    sizes = [r['num_items'] for r in results['cuckoo']]
    
    ax1.plot(sizes, [r['memory_mb'] for r in results['cuckoo']], 'b-', label='Cuckoo')
    ax1.plot(sizes, [r['memory_mb'] for r in results['xor']], 'r--', label='XOR')
    ax1.set_title('Memory Usage')
    ax1.set_xlabel('Number of Items')
    ax1.set_ylabel('Memory (MB)')
    ax1.legend()
    
    ax2.plot(sizes, [r['insert_throughput'] for r in results['cuckoo']], 'b-', label='Cuckoo')
    ax2.plot(sizes, [r['insert_throughput'] for r in results['xor']], 'r--', label='XOR')
    ax2.set_title('Insert Throughput')
    ax2.set_xlabel('Number of Items')
    ax2.set_ylabel('Items/second')
    ax2.legend()
    
    ax3.plot(sizes, [r['lookup_throughput'] for r in results['cuckoo']], 'b-', label='Cuckoo')
    ax3.plot(sizes, [r['lookup_throughput'] for r in results['xor']], 'r--', label='XOR')
    ax3.set_title('Lookup Throughput')
    ax3.set_xlabel('Number of Items')
    ax3.set_ylabel('Lookups/second')
    ax3.legend()
    
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