#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <tuple>
#include <ATen/ATen.h>
#include "quant_cuda.h"
#include "quant_kernel.h"
#include <iostream>

using namespace at;

Tensor get_max_entry(Tensor a, int dim) {
  Tensor max_entry;
  if (dim == -1) {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  } else if (dim == 0) {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.abs().max(1, true)).expand_as(input_view).view_as(a).contiguous();
  } else {
    Tensor input_transpose = a.transpose(0, dim);
    Tensor input_view = input_transpose.contiguous().view({input_transpose.size(0), -1});
    Tensor max_transpose = std::get<0>(input_view.abs().max(1, true)).expand_as(input_view).view_as(input_transpose);
    max_entry = max_transpose.transpose(dim, 0).contiguous();
  }
  return max_entry;
}

Tensor block_quantize_stochastic_cuda(Tensor a, int wl, int dim) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = get_max_entry(a, dim);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_stochastic<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                    rand_ints.data_ptr<int>(),
                                                    o.data_ptr<float>(),
                                                    size,
                                                    max_entry.data_ptr<float>(),
                                                    wl);
  return o;
}

Tensor block_quantize_nearest_cuda(Tensor a, int wl, int dim) {
  auto o = at::zeros_like(a);
  int64_t size = a.numel();

  Tensor max_entry = get_max_entry(a, dim);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_nearest<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                 o.data_ptr<float>(),
                                                 size,
                                                 max_entry.data_ptr<float>(),
                                                 wl);
  return o;
}

Tensor block_quantize_sim_stochastic_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_stochastic<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                        rand_probs.data_ptr<float>(),
                                                        o.data_ptr<float>(),
                                                        size,
                                                        max_entry.data_ptr<float>(),
                                                        wl);
  return o;
}

Tensor block_quantize_sim_nearest_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_nearest<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                 o.data_ptr<float>(),
                                                 size,
                                                 max_entry.data_ptr<float>(),
                                                 wl);
  return o;
}

Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_stochastic<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                    rand_ints.data_ptr<int>(),
                                                    o.data_ptr<float>(),
                                                    size,
                                                    man_bits,
                                                    exp_bits);
  return o;
}

Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                 o.data_ptr<float>(),
                                                 size,
                                                 man_bits,
                                                 exp_bits);
  return o;
}

void fixed_min_max(int wl, int fl, bool symmetric, float* t_min, float* t_max) {
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl-fl-1);
  *t_max = -*t_min-ldexp(1.0, sigma);
  if (symmetric) *t_min = *t_min+ldexp(1.0, sigma);
}

Tensor fixed_point_quantize_stochastic_cuda(Tensor a, int wl, int fl, bool use_clamp, bool symmetric) {
  // use external random number right now
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_stochastic<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                                   rand_probs.data_ptr<float>(),
                                                                   o.data_ptr<float>(),
                                                                   size,
                                                                   sigma,
                                                                   use_clamp,
                                                                   t_min,
                                                                   t_max);
  return o;
}

Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl, bool use_clamp, bool symmetric) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_nearest<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                                o.data_ptr<float>(),
                                                                size,
                                                                sigma,
                                                                use_clamp,
                                                                t_min,
                                                                t_max);
  return o;
}


void backtrack(std::string& binary, int index, int length, std::vector<float>& result) {
    if (index == length) {
        // std::cout << "Binary: " << binary << std::endl; 
        result.push_back(std::stoi(binary, nullptr, 2));
        return;
    }
    
    // Append '0' and continue the recursion
    binary[index] = '0';
    backtrack(binary, index + 1, length, result);
    
    // If the last digit appended is '0', then we can append '1' as well
    if (index == 0 || binary[index - 1] == '0') {
        binary[index] = '1';
        backtrack(binary, index + 1, length, result);
    }
}

std::vector<float> generateNonAdjacentOnesSequences(int length) {
    std::vector<float> result;
    string binary(length, ' '); // Initialize a string of length 'length' filled with spaces
    backtrack(binary, 0, length, result);
    return result;
}

Tensor noAdjecentOnes_quantize_nearest_cuda(Tensor a, int wl, int fl, bool use_clamp, bool symmetric) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

    // Generate Fibonacci sequence up to 2^wl
    std::vector<float> codewords = generateNonAdjacentOnesSequences(wl);

      for (auto& num : codewords) {
      num /= (1 << fl);
      // num = num - pow(2, wl - fl - 1); 
  }

    // std::cout << "Codewords: " << std::endl;
    // for (int i = 0; i < codewords.size(); i++) {
    //     std::cout << codewords[i] << std::endl;
    // }

    // printf("Codewords size: %d\n", codewords.size());

    // Convert codewords to array
    float* d_codewords;
    cudaMalloc(&d_codewords, codewords.size() * sizeof(float));
    cudaMemcpy(d_codewords, codewords.data(), codewords.size() * sizeof(float), cudaMemcpyHostToDevice);


  noAdjecentOnes_quantize_kernel_nearest<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                                o.data_ptr<float>(),
                                                                size,
                                                                d_codewords,
                                                                codewords.size(),
                                                                t_min,
                                                                t_max);
  return o;
}

std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask_cuda(Tensor a, int wl, int fl, bool symmetric) {
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_probs = rand_like(a);
  auto m = zeros_like(a, a.options().dtype(kByte));
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_mask_stochastic<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                                        rand_probs.data_ptr<float>(),
                                                                        o.data_ptr<float>(),
                                                                        m.data_ptr<uint8_t>(),
                                                                        size,
                                                                        sigma,
                                                                        t_min,
                                                                        t_max);
  return std::make_tuple(o, m);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_nearest_mask_cuda(Tensor a, int wl, int fl, bool symmetric) {
  // use external random number right now
  auto o = at::zeros_like(a);
  auto m = zeros_like(a, a.options().dtype(kByte));
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_mask_nearest<<<blockNums, blockSize>>>(a.data_ptr<float>(),
                                                                     o.data_ptr<float>(),
                                                                     m.data_ptr<uint8_t>(),
                                                                     size,
                                                                     sigma,
                                                                     t_min,
                                                                     t_max);
  return std::make_tuple(o, m);
}