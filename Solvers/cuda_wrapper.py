import torch
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
import torch.cuda as cuda
from torch.utils.cpp_extension import load_inline
import os

class ParallelFunction:
    def __init__(self, fct: Callable, num_workers: int = None):
        """
        Parallelizes a function that operates on single (non-batched) inputs.
        
        Args:
            fct: Function that takes a single tensor of shape `dim` as input (non-batched)
            num_workers: Number of parallel workers (for thread method)
        """
        self.fct = fct
        self.num_workers = num_workers or max(1, os.cpu_count())
        
        if torch.cuda.is_available(): 
            self._setup_cuda_kernel()
    
    def _setup_cuda_kernel(self):
        """
        Creates a CUDA kernel wrapper that can call our Python function in parallel.
        This is a simplified version - a full implementation would need more sophisticated
        Python/CUDA interop.
        """
        cuda_wrapper = """
        #include <torch/extension.h>
        
        std::vector<torch::Tensor> parallel_wrapper(
            torch::Tensor input,
            py::function python_func) {
            
            const int batch_size = input.size(0);
            std::vector<torch::Tensor> results;
            results.reserve(batch_size);
            
            // Process each input in parallel using CUDA streams
            std::vector<torch::cuda::CUDAStream> streams;
            const int num_streams = std::min(batch_size, 
                torch::cuda::device_count() * 4);
            
            for (int i = 0; i < num_streams; ++i) {
                streams.push_back(torch::cuda::getStreamFromPool());
            }
            
            for (int i = 0; i < batch_size; ++i) {
                const auto stream = streams[i % num_streams];
                torch::cuda::CUDAStreamGuard guard(stream);
                
                // Extract single input
                auto single_input = input.select(0, i);
                
                // Call Python function and store result
                py::gil_scoped_acquire acquire;
                auto result = python_func(single_input);
                results.push_back(result.cast<torch::Tensor>());
            }
            
            // Synchronize all streams
            for (const auto& stream : streams) {
                stream.synchronize();
            }
            
            return results;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("parallel_wrapper", &parallel_wrapper, 
                "Parallel wrapper for Python function");
        }
        """
        
        # Compile the CUDA wrapper
        self.cuda_module = load_inline(
            name='parallel_wrapper',
            cpp_sources=cuda_wrapper,
            functions=['parallel_wrapper'],
            with_cuda=True,
            extra_cuda_cflags=['-O2']
        )
    
    def __call__(self, batch_input: torch.Tensor) -> torch.Tensor:
        device = batch_input.device 
        if device == torch.device("cpu"):
            return self._parallel_cpu(batch_input)
        elif device == torch.device("cuda"):
            return self._parallel_cuda(batch_input)
        else:
            raise ValueError(f"Unknown device: {device}")
    
    def _parallel_cpu(self, batch_input: torch.Tensor) -> torch.Tensor:
        """
        Thread-based CPU parallelization for single-input functions.
        """
        batch_size = batch_input.shape[0]
        
        def process_single(idx):
            single_input = batch_input[idx].contiguous()
            return self.fct(single_input)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_single, range(batch_size)))
            
        return torch.stack(results)
    
    def _parallel_cuda(self, batch_input: torch.Tensor) -> torch.Tensor:
        """
        CUDA parallelization for single-input functions using multiple streams.
        """
        if not batch_input.is_cuda:
            batch_input = batch_input.cuda()
        
        # Use our compiled CUDA wrapper to parallelize the function
        results = self.cuda_module.parallel_wrapper(batch_input, self.fct)
        return torch.stack(results)

# Example usage
def example():
    # Define a function that only works on single inputs
    def single_input_function(x: torch.Tensor) -> torch.Tensor:
        # This function can only handle non-batched inputs
        assert len(x.shape) == 1, "Function only works on single inputs!"
        return x.pow(2) + 2 * x + 1

    # Create parallel version
    parallel_fn = ParallelFunction(single_input_function)

    # Use with batched input
    batch = torch.randn(1000, 10)  # batch_size=1000, dim=10
    result = parallel_fn(batch)  # Will process each input in parallel
    
    return result