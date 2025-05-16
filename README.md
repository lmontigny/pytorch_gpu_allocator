# Custom PyTorch GPU Memory Allocator with OOM Handling

This extension provides a custom CUDA memory allocator for PyTorch that intercepts out-of-memory (OOM) errors and handles them gracefully instead of crashing. It implements an intelligent retry mechanism with exponential backoff and size reduction.

## Features

- Intercepts CUDA memory allocation attempts and handles OOM conditions
- Automatically retries allocations with reduced size when OOM occurs
- Implements exponential backoff to wait for memory to become available
- Configurable via environment variables
- Minimal overhead during normal operation
- Simple API for tensor allocation with OOM handling
- Comprehensive memory statistics reporting

## Installation
On Lightning AI (https://lightning.ai/) using T4 GPU studio environment.
```bash
python setup.py install
```

## Usage

```python
import torch
import custom_cuda_allocator

# Initialize the allocator
custom_cuda_allocator.setup_allocator()

# Get memory statistics
stats = custom_cuda_allocator.get_memory_stats()
print(f"GPU memory: {stats['used_gb']:.2f} GB used / {stats['total_gb']:.2f} GB total")

# Allocate tensor with OOM protection
tensor = custom_cuda_allocator.allocate_tensor(
    num_elements=1000000000,  # 1 billion elements
    dtype=torch.float32,
    device=0
)

# Use the tensor normally
tensor.fill_(1.0)

# Clear cache when done
custom_cuda_allocator.clear_cache()
```

## Configuration

The allocator can be configured via environment variables:

- `OOM_MAX_RETRIES`: Maximum number of retry attempts (default: 3)
- `OOM_SIZE_FACTOR`: Size reduction factor for each retry (default: 0.7)
- `OOM_DELAY_MS`: Base delay for exponential backoff in milliseconds (default: 200)
- `OOM_VERBOSE`: Enable verbose logging (default: 0, set to 1 to enable)

Example:
```bash
export OOM_MAX_RETRIES=5
export OOM_SIZE_FACTOR=0.8
export OOM_DELAY_MS=100
export OOM_VERBOSE=1
python your_script.py
```

## API Reference

### Main Functions

- `setup_allocator()`: Initialize the OOM-handling allocator
- `allocate_tensor(elements, dtype=torch.float32, device=0)`: Allocate a tensor with OOM handling
- `get_memory_stats(device=-1)`: Get GPU memory statistics
- `clear_cache()`: Clear CUDA memory cache
- `get_retry_count()`: Get number of allocation retries

### CUDAAllocator Class

For advanced usage, you can access the underlying allocator class:

```python
from custom_cuda_allocator import CUDAAllocator

# Create allocator instance
allocator = CUDAAllocator()

# Direct memory allocation
ptr = allocator.allocate(size_in_bytes, device=0)
allocator.deallocate(ptr)

# Create tensor
tensor = allocator.create_tensor(num_elements, torch.TensorOptions().dtype(torch.float32).device(torch.kCUDA))
```

## Testing

Run the included test script to verify functionality:

```bash
python test.py
```

The test script verifies:
1. Direct allocation functionality
2. OOM handling and retry mechanism
3. Exponential backoff behavior
4. Memory recovery after OOM conditions

## License

BSD