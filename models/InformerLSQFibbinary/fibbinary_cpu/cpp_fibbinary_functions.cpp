#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Function to find the closest Fibonacci binary number to a given value
int closest_fibbinary(float val) {
    // List of Fibonacci binary numbers up to a certain limit
    std::vector<int> fibbinary = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597};

    // Find the closest Fibonacci binary number to the given value
    int closest = *std::min_element(fibbinary.begin(), fibbinary.end(), [&](int a, int b) {
        return std::abs(a - val) < std::abs(b - val);
    });

    return closest;
}

// Function to find the closest Fibonacci binary numbers for a 2D tensor of floats
torch::Tensor closest_fibbinary_array_2d(torch::Tensor values) {
    // Check if the tensor is non-empty and has 2 dimensions
    TORCH_CHECK(values.dim() == 2 && values.size(1) > 0, "Input tensor must be a 2D tensor with at least one element");

    // Accessor for 2D tensor
    auto values_accessor = values.accessor<float, 2>();

    // Vector to store results
    std::vector<std::vector<int>> closest_fibs(values.size(0), std::vector<int>(values.size(1)));

    // Iterate over the tensor elements
    for (int i = 0; i < values.size(0); ++i) {
        for (int j = 0; j < values.size(1); ++j) {
            float val = values_accessor[i][j];
            closest_fibs[i][j] = closest_fibbinary(val);
        }
    }

    // Convert the vector of vectors to a tensor
    torch::Tensor result = torch::from_blob(closest_fibs.data(), {values.size(0), values.size(1)}, torch::dtype(torch::kInt32));

    return result;
}

// Function to find the closest Fibonacci binary numbers for a 3D tensor of floats
torch::Tensor closest_fibbinary_array_3d(torch::Tensor values) {
    // Check if the tensor is non-empty and has 3 dimensions
    TORCH_CHECK(values.dim() == 3 && values.size(1) > 0 && values.size(2) > 0, "Input tensor must be a 3D tensor with at least one element in each dimension");

    // Accessor for 3D tensor
    auto values_accessor = values.accessor<float, 3>();

    // Vector to store results
    std::vector<std::vector<std::vector<int>>> closest_fibs(values.size(0), std::vector<std::vector<int>>(values.size(1), std::vector<int>(values.size(2))));

    // Iterate over the tensor elements
    for (int i = 0; i < values.size(0); ++i) {
        for (int j = 0; j < values.size(1); ++j) {
            for (int k = 0; k < values.size(2); ++k) {
                float val = values_accessor[i][j][k];
                closest_fibs[i][j][k] = closest_fibbinary(val);
            }
        }
    }

    // Convert the vector of vectors to a tensor
    torch::Tensor result = torch::from_blob(closest_fibs.data(), {values.size(0), values.size(1), values.size(2)}, torch::dtype(torch::kInt32));

    return result;
}

// Binding function for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("closest_fibbinary_array_2d", &closest_fibbinary_array_2d, "Find the closest Fibonacci binary numbers for a 2D array of floats");
    m.def("closest_fibbinary_array_3d", &closest_fibbinary_array_3d, "Find the closest Fibonacci binary numbers for a 3D array of floats");
}
