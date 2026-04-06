import time
from functools import wraps

PROFILER_DATA = {}

def profile(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            dt = (end - start) * 1000.0  # ms
            
            if name not in PROFILER_DATA:
                PROFILER_DATA[name] = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
            
            stats = PROFILER_DATA[name]
            stats["count"] += 1
            stats["total_ms"] += dt
            stats["max_ms"] = max(stats["max_ms"], dt)
            
            return result
        return wrapper
    return decorator

def print_stats():
    print("\n[Profiler Stats]")
    print(f"{'Name':<20} | {'Avg (ms)':<10} | {'Max (ms)':<10} | {'Count'}")
    print("-" * 55)
    for name, stats in PROFILER_DATA.items():
        avg = stats["total_ms"] / stats["count"]
        print(f"{name:<20} | {avg:<10.2f} | {stats['max_ms']:<10.2f} | {stats['count']}")
    print("-" * 55)
