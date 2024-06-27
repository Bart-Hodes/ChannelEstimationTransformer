#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>

// CUDA kernel to find the closest Fibonacci binary number to a given value for a 2D tensor
__global__ void closest_fibbinary_kernel_2d(const float *values, int *closest_fibs, int rows, int cols, const int* fibbinary, int fibbinary_size)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (index_x < rows && index_y < cols)
    {
        float val = values[index_x * cols + index_y];
        closest_fibs[index_x * cols + index_y] = fibbinary[0];
        for (int i = 1; i < fibbinary_size; ++i)
        {
            if (std::abs(fibbinary[i] - val) < std::abs(closest_fibs[index_x * cols + index_y] - val))
            {
                closest_fibs[index_x * cols + index_y] = fibbinary[i];
            }
        }
    }
}

// CUDA kernel to find the closest Fibonacci binary number to a given value for a 3D tensor
__global__ void closest_fibbinary_kernel_3d(const float *values, int *closest_fibs, int dim0, int dim1, int dim2, const int* fibbinary, int fibbinary_size)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int index_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (index_x < dim0 && index_y < dim1 && index_z < dim2)
    {
        float val = values[(index_x * dim1 + index_y) * dim2 + index_z];
        closest_fibs[(index_x * dim1 + index_y) * dim2 + index_z] = fibbinary[0];
        for (int i = 1; i < fibbinary_size; ++i)
        {
            if (std::abs(fibbinary[i] - val) < std::abs(closest_fibs[(index_x * dim1 + index_y) * dim2 + index_z] - val))
            {
                closest_fibs[(index_x * dim1 + index_y) * dim2 + index_z] = fibbinary[i];
            }
        }
    }
}

// Function to find the closest Fibonacci binary numbers for a 2D tensor of floats on GPU
torch::Tensor closest_fibbinary_array_2d(torch::Tensor values, torch::Tensor fibbinary)
{
    // Check if the tensor is non-empty and has 2 dimensions
    TORCH_CHECK(values.dim() == 2 && values.size(0) > 0 && values.size(1) > 0, "Input tensor must be a 2D tensor with at least one element in each dimension");

    // Allocate memory for result tensor
    torch::Tensor closest_fibs = torch::empty_like(values, torch::dtype(torch::kInt32));

    // Allocate device memory for fibbinary
    int* fibbinary_device;
    cudaMalloc((void**)&fibbinary_device, fibbinary.size(0) * sizeof(int));

    // Copy fibbinary from host to device
    cudaMemcpy(fibbinary_device, fibbinary.data_ptr<int>(), fibbinary.size(0) * sizeof(int), cudaMemcpyHostToDevice);


    // Launch CUDA kernel for 2D tensor
    const int threads_per_block_x = 16;
    const int threads_per_block_y = 16;
    const dim3 threads_per_block(threads_per_block_x, threads_per_block_y);
    const dim3 num_blocks((values.size(0) + threads_per_block_x - 1) / threads_per_block_x, (values.size(1) + threads_per_block_y - 1) / threads_per_block_y);
    closest_fibbinary_kernel_2d<<<num_blocks, threads_per_block>>>(
        values.data_ptr<float>(), closest_fibs.data_ptr<int>(), values.size(0), values.size(1), fibbinary_device, fibbinary.size(0)
        );

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    return closest_fibs;
}

// Function to find the closest Fibonacci binary numbers for a 3D tensor of floats on GPU
torch::Tensor closest_fibbinary_array_3d(torch::Tensor values, torch::Tensor fibbinary)
{
    // Check if the tensor is non-empty and has 3 dimensions
    TORCH_CHECK(values.dim() == 3 && values.size(0) > 0 && values.size(1) > 0 && values.size(2) > 0, "Input tensor must be a 3D tensor with at least one element in each dimension");

    // Allocate memory for result tensor
    torch::Tensor closest_fibs = torch::empty_like(values, torch::dtype(torch::kInt32));

    // Allocate device memory for fibbinary
    int* fibbinary_device;
    cudaMalloc((void**)&fibbinary_device, fibbinary.size(0) * sizeof(int));

    // Copy fibbinary from host to device
    cudaMemcpy(fibbinary_device, fibbinary.data_ptr<int>(), fibbinary.size(0) * sizeof(int), cudaMemcpyHostToDevice);


    // Launch CUDA kernel for 3D tensor
    const int threads_per_block_x = 8;
    const int threads_per_block_y = 8;
    const int threads_per_block_z = 8;
    const dim3 threads_per_block(threads_per_block_x, threads_per_block_y, threads_per_block_z);
    const dim3 num_blocks((values.size(0) + threads_per_block_x - 1) / threads_per_block_x, (values.size(1) + threads_per_block_y - 1) / threads_per_block_y, (values.size(2) + threads_per_block_z - 1) / threads_per_block_z);
    closest_fibbinary_kernel_3d<<<num_blocks, threads_per_block>>>(
        values.data_ptr<float>(), closest_fibs.data_ptr<int>(), values.size(0), values.size(1), values.size(2), fibbinary_device, fibbinary.size(0)
        );

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    return closest_fibs;
}

// Binding function for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("closest_fibbinary_array_2d", &closest_fibbinary_array_2d, "Find the closest Fibonacci binary numbers for a 2D array of floats on GPU");
    m.def("closest_fibbinary_array_3d", &closest_fibbinary_array_3d, "Find the closest Fibonacci binary numbers for a 3D array of floats on GPU");
}
