#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__ 
void dijkstra_grid_kernel(float* weight_tens, float* solution, int b, int h, int w) {
    int idx = blockIdx.x; // Each block handles a different grid
    if (idx >= b) return; // Check out-of-bounds

    extern __shared__ float shared_mem[]; // Shared memory allocation
    float* distances = shared_mem;        // Distance map for this grid
    bool* visited = (bool*)&distances[h * w]; // Visited map following distances
    int* predecessor = (int*)&visited[h * w]; // Predecessors following visited
    
    // Initialize variables
    int initial_node = 0;       // (0,0) flattened
    int desired_node = h * w - 1; // (h-1, w-1) flattened
    
    for (int i = threadIdx.x; i < h * w; i += blockDim.x) {
        distances[i] = INFINITY;
        visited[i] = false;
    }
    __syncthreads();

    // Source node has a distance of 0 to itself
    distances[initial_node] = 0.0f;

    int current_node = initial_node;

    while (true) {
        int x = current_node / w;
        int y = current_node % w;

        __syncthreads();
        if (threadIdx.x == 0) visited[current_node] = true;
        __syncthreads();

        // Process neighbors
        int directions[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, 
                                {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        for (int i = 0; i < 8; ++i) {
            int newX = x + directions[i][0];
            int newY = y + directions[i][1];
            if (newX >= 0 && newX < h && newY >= 0 && newY < w) {
                int neighbor = newX * w + newY;
                if (!visited[neighbor]) {
                    float newDist = distances[current_node] + weight_tens[idx * h * w + neighbor];
                    if (newDist < distances[neighbor]) {
                        distances[neighbor] = newDist;
                        predecessor[neighbor * 2] = x;
                        predecessor[neighbor * 2 + 1] = y;
                    }
                }
            }
        }
        __syncthreads();

        // Select the next node with the smallest distance
        float min_dist = INFINITY;
        int next_node = -1;
        for (int i = threadIdx.x; i < h * w; i += blockDim.x) {
            if (!visited[i] && distances[i] < min_dist) {
                min_dist = distances[i];
                next_node = i;
            }
        }

        current_node = next_node;
        __syncthreads();

        if (current_node == desired_node || current_node == -1) break;
    }

    // Backtrack to create the path
    int path_node = desired_node;
    while (path_node != initial_node) {
        int px = predecessor[path_node * 2];
        int py = predecessor[path_node * 2 + 1];
        int prev_node = px * w + py;
        if (threadIdx.x == 0) solution[idx * h * w + path_node] = 1.0f;
        path_node = prev_node;
        __syncthreads();
    }
}


torch::Tensor dijkstra_grid(torch::Tensor weight_tens) {
    // Step 1: Allocate memory on GPU and copy contents from Host to Device
    int batch_size = weight_tens.size(0); 
    int height = weight_tens.size(1);      
    int width = weight_tens.size(2);      

    size_t size = batch_size * height * width * sizeof(float);

    // Allocate GPU memory for the input and output tensors
    float* weight_tens_dev;
    cudaMalloc((void**)&weight_tens_dev, size);
    cudaMemcpy(weight_tens_dev, weight_tens.data_ptr<float>(), size, cudaMemcpyHostToDevice);

    // Allocate memory for the output tensor (solution on device)
    torch::Tensor solution = torch::empty({batch_size, height, width}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    float* solution_dev;
    cudaMalloc((void**)&solution_dev, size);

    // Step 2: Launch CUDA kernel
    dim3 blockSize(256);
    dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x);

    // Launch the kernel for each 2D grid in the batch
    dijkstra_grid_kernel<<<gridSize, blockSize>>>(weight_tens_dev, solution_dev, batch_size, height, width);

    // Step 3: Copy the solution from Device to Host
    cudaMemcpy(solution.data_ptr<float>(), solution_dev, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(weight_tens_dev);
    cudaFree(solution_dev);

    return solution;
}
