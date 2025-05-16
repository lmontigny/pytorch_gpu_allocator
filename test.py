#!/usr/bin/env python3
"""Test suite for custom CUDA allocator with OOM handling."""

import os
import time
import torch
import custom_cuda_allocator
from custom_cuda_allocator import OOMAllocator  # Import the OOMAllocator class

# Configure allocator via environment variables
os.environ["OOM_MAX_RETRIES"] = "3"
os.environ["OOM_SIZE_FACTOR"] = "0.7"
os.environ["OOM_DELAY_MS"] = "200"
os.environ["OOM_VERBOSE"] = "1"

def print_gpu_info():
    """Print GPU information and memory stats."""
    print("\n=== GPU Information ===")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name} ({props.total_memory / (1024**3):.2f} GB)")
    
    stats = custom_cuda_allocator.get_memory_stats()
    print(f"Memory: {stats['used_gb']:.2f} GB used / {stats['total_gb']:.2f} GB total")

def test_normal_allocation():
    """Test normal tensor allocation."""
    print("\n=== Testing Normal Allocation ===")
    stats = custom_cuda_allocator.get_memory_stats()
    size_gb = stats['free_gb'] * 0.5  # Use 50% of free memory
    elements = int(size_gb * (1024**3) / 4)  # Assuming float32
    
    print(f"Allocating {size_gb:.2f} GB tensor...")
    tensor = custom_cuda_allocator.allocate_tensor(elements)
    
    print(f"Success! Tensor shape: {tensor.shape}")
    tensor.fill_(1.0)  # Test usability
    print(f"Tensor is usable (first value: {tensor[0].item()})")
    
    del tensor
    custom_cuda_allocator.clear_cache()
    return True

def test_raw_allocation():
    """Test raw memory allocation using OOMAllocator directly."""
    print("\n=== Testing Raw Memory Allocation ===")
    stats = custom_cuda_allocator.get_memory_stats()
    size_bytes = int(stats['free_gb'] * 0.3 * (1024**3))  # Use 30% of free memory
    
    print(f"Allocating {size_bytes/(1024**3):.2f} GB of raw memory...")
    
    # Create allocator instance
    allocator = OOMAllocator()
    
    # Allocate raw memory
    try:
        ptr = allocator.allocate(size_bytes)
        print(f"Success! Raw memory allocated at {ptr}")
        
        # Deallocate
        allocator.deallocate(ptr)
        print("Memory successfully deallocated")
        
        # Clear cache to ensure memory is fully released
        custom_cuda_allocator.clear_cache()
        print("Cache cleared")
        
        return True
    except Exception as e:
        print(f"Raw allocation failed: {e}")
        return False

def test_oom_handling():
    """Test OOM handling with retries and backoff."""
    print("\n=== Testing OOM Handling ===")
    stats = custom_cuda_allocator.get_memory_stats()
    
    # Step 1: Fill most of memory
    fill_size_gb = stats['free_gb'] * 0.8
    fill_elements = int(fill_size_gb * (1024**3) / 4)
    print(f"1. Filling {fill_size_gb:.2f} GB of memory...")
    fill_tensor = custom_cuda_allocator.allocate_tensor(fill_elements)
    
    # Step 2: Try allocating more than remaining memory
    print("\n2. Attempting oversized allocation (should trigger OOM)...")
    oom_size_gb = stats['free_gb'] * 0.5  # This should exceed available memory
    oom_elements = int(oom_size_gb * (1024**3) / 4)
    
    start_time = time.time()
    retry_tensor = custom_cuda_allocator.allocate_tensor(oom_elements)
    elapsed = time.time() - start_time
    
    if retry_tensor is not None:
        actual_gb = retry_tensor.numel() * 4 / (1024**3)
        reduction = (1 - actual_gb/oom_size_gb) * 100
        
        print(f"\nOOM handling succeeded!")
        print(f"- Requested: {oom_size_gb:.2f} GB")
        print(f"- Allocated: {actual_gb:.2f} GB ({reduction:.1f}% reduction)")
        print(f"- Time with backoff: {elapsed:.2f} seconds")
        
        del retry_tensor
    else:
        print("Failed to allocate after retries")
    
    del fill_tensor
    custom_cuda_allocator.clear_cache()
    
    # Verify retry count increased
    retries = custom_cuda_allocator.get_retry_count()
    print(f"Total retries: {retries}")
    
    return retries > 0

def main():
    """Run the test suite."""
    print("Testing Custom CUDA Allocator with OOM Handling")
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return 1
    
    # Initialize allocator
    custom_cuda_allocator.setup_allocator()
    print_gpu_info()
    
    # Run tests and collect results
    results = {
        "Normal allocation": test_normal_allocation(),
        "Raw memory allocation": test_raw_allocation(),
        "OOM handling": test_oom_handling()
    }
    
    # Small allocation after OOM tests
    try:
        t = torch.ones(1000, 1000, device='cuda')
        results["Post-OOM allocation"] = True
        del t
    except Exception as e:
        print(f"Post-OOM allocation failed: {e}")
        results["Post-OOM allocation"] = False
    
    # Print results
    print("\n=== Test Results ===")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        all_passed = all_passed and passed
    
    print(f"\nOverall: {'✅ PASS' if all_passed else '❌ FAIL'}")
    print("\n=== Final Memory State ===")
    print_gpu_info()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())