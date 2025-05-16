#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <memory>

// Global stats and config
static std::atomic<int> g_retry_count{0};
static int g_max_retries = std::getenv("OOM_MAX_RETRIES") ? std::atoi(std::getenv("OOM_MAX_RETRIES")) : 3;
static double g_size_factor = std::getenv("OOM_SIZE_FACTOR") ? std::atof(std::getenv("OOM_SIZE_FACTOR")) : 0.7;
static int g_retry_delay_ms = std::getenv("OOM_DELAY_MS") ? std::atoi(std::getenv("OOM_DELAY_MS")) : 200;
static bool g_verbose = std::getenv("OOM_VERBOSE") ? (std::atoi(std::getenv("OOM_VERBOSE")) != 0) : false;

#define OOM_LOG(msg) if (g_verbose) std::cout << "[OOM] " << msg << std::endl

class OOMAllocator {
public:
    OOMAllocator() { 
        OOM_LOG("Allocator created");
        cudaFree(0); // Init CUDA context
    }
    
    // Memory allocation with OOM retry
    void* allocate(size_t size, int device = 0) {
        if (device >= 0) cudaSetDevice(device);
        void* ptr = nullptr;
        size_t current_size = size;
        
        for (int attempt = 0; attempt < g_max_retries; attempt++) {
            try {
                ptr = c10::cuda::CUDACachingAllocator::raw_alloc(current_size);
                break;
            } catch (const c10::Error&) {
                g_retry_count++;
                current_size = static_cast<size_t>(current_size * g_size_factor);
                int backoff_ms = g_retry_delay_ms * (1 << std::min(attempt, 4));
                
                OOM_LOG("Retry " << attempt+1 << "/" << g_max_retries 
                      << ", size: " << current_size << ", backoff: " << backoff_ms << "ms");
                
                if (attempt + 1 == g_max_retries) throw;
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            }
        }
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (ptr) c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }
    
    // Create tensor with OOM retry
    torch::Tensor create_tensor(size_t num_elements, torch::TensorOptions options) {
        OOM_LOG("Creating tensor with " << num_elements << " elements");
        int device = options.device().index();
        if (device >= 0) cudaSetDevice(device);
        
        try {
            return torch::empty({static_cast<int64_t>(num_elements)}, options);
        } catch (const c10::Error&) {
            for (int attempt = 0; attempt < g_max_retries; attempt++) {
                try {
                    size_t adjusted = static_cast<size_t>(num_elements * std::pow(g_size_factor, attempt+1));
                    int backoff_ms = g_retry_delay_ms * (1 << std::min(attempt, 4));
                    
                    OOM_LOG("Tensor retry with " << adjusted << " elements");
                    std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
                    
                    return torch::empty({static_cast<int64_t>(adjusted)}, options);
                } catch (const c10::Error&) {
                    g_retry_count++;
                    if (attempt + 1 == g_max_retries) throw;
                }
            }
        }
        return torch::Tensor();
    }
    
    bool clear_cache() {
        try {
            c10::cuda::CUDACachingAllocator::emptyCache();
            return true;
        } catch (...) {
            return false;
        }
    }
};

// Global allocator
std::shared_ptr<OOMAllocator> g_allocator;

// Get memory stats
py::dict get_memory_stats(int device = -1) {
    if (device >= 0) cudaSetDevice(device);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return py::dict(
        py::arg("free_bytes") = free,
        py::arg("total_bytes") = total, 
        py::arg("used_bytes") = total - free,
        py::arg("free_gb") = static_cast<double>(free) / (1024 * 1024 * 1024),
        py::arg("total_gb") = static_cast<double>(total) / (1024 * 1024 * 1024),
        py::arg("used_gb") = static_cast<double>(total - free) / (1024 * 1024 * 1024)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Export allocator class
    py::class_<OOMAllocator, std::shared_ptr<OOMAllocator>>(m, "OOMAllocator")
        .def(py::init<>())
        .def("allocate", &OOMAllocator::allocate, py::arg("size"), py::arg("device") = 0)
        .def("deallocate", &OOMAllocator::deallocate)
        .def("create_tensor", &OOMAllocator::create_tensor)
        .def("clear_cache", &OOMAllocator::clear_cache);

    // Helper functions
    m.def("setup_allocator", []() {
        if (!g_allocator) g_allocator = std::make_shared<OOMAllocator>();
        return true;
    });
    
    m.def("allocate_tensor", [](size_t elements, torch::ScalarType dtype, int device) {
        if (!g_allocator) g_allocator = std::make_shared<OOMAllocator>();
        try {
            return g_allocator->create_tensor(elements, 
                torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device));
        } catch (...) {
            return torch::Tensor();
        }
    }, py::arg("elements"), py::arg("dtype") = torch::kFloat32, py::arg("device") = 0);
    
    m.def("get_memory_stats", &get_memory_stats, py::arg("device") = -1);
    
    m.def("clear_cache", []() {
        if (!g_allocator) g_allocator = std::make_shared<OOMAllocator>();
        return g_allocator->clear_cache();
    });
    
    m.def("get_retry_count", []() { return g_retry_count.load(); });
}