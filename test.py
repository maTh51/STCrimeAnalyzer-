import time 
import psutil
import gpustat

def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


memory_usage = psutil.virtual_memory()
print(f"Memory Usage before: {memory_usage.percent}%")


gpu_stats_start = gpustat.GPUStatCollection.new_query()

start = time.time()
fibonacci(20)
end = time.time()

gpu_stats_end = gpustat.GPUStatCollection.new_query()

print(gpu_stats_start)

for gpu_start, gpu_end in zip(gpu_stats_start.gpus, gpu_stats_end):
    print(f"GPU {gpu_start.index}: {gpu_start.name}, Utilization: {gpu_end.utilization - gpu_start.utilization}%")


memory_usage = psutil.virtual_memory()
print(f"Memory Usage after: {memory_usage.percent}%")


elapsed = end - start
print(f'Time taken: {elapsed:.6f} seconds')