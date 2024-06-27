#include "quant_kernel.h"
#include "sim_helper.cu"

#include <stdio.h>


template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

template <typename T>
__device__ __forceinline__ T clamp_mask_helper(T a, T min, T max, uint8_t* mask) {
  if (a > max) {
    *mask = 1;
    return max;
  } else if (a < min) {
    *mask = 1;
    return min;
  }
  *mask = 0;
  return a;
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Stochastic Rounding with r.
__global__ void fixed_point_quantize_kernel_stochastic(float* __restrict__ a,
                                                       float* __restrict__ r,
                                                       float* o, int size,
                                                       int sigma, bool use_clamp,
                                                       float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Nearest Neighbor Rounding.
__global__ void fixed_point_quantize_kernel_nearest(float* __restrict__ a,
                                                    float* o, int size,
                                                    int sigma, bool use_clamp,
                                                    float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}


__device__ float findNearestNumber(float n, const float* fibonacci, int size) {
    float nearest = fibonacci[0];
    for (int i = 1; i < size; ++i) {
        if (fabs(n - fibonacci[i]) < fabs(n - nearest)) {
            nearest = fibonacci[i];
        }
    }
    return nearest;
}

__global__ void noAdjecentOnes_quantize_kernel_nearest(const float* a, float* o, int size, const float* codewords, int codewordSize, float t_min, float t_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float value = a[tid];
        float nearest_num = findNearestNumber(value, codewords, codewordSize);
        o[tid] = nearest_num;
        // Clamp if necessary
        if (o[tid] < t_min) {
            o[tid] = t_min;
        }
        else if (o[tid] > t_max) {
            o[tid] = t_max;
        }
    }
}



__global__ void fixed_point_quantize_kernel_mask_stochastic(float* __restrict__ a,
                                                            float* __restrict__ r,
                                                            float* o, uint8_t* m,
                                                            int size, int sigma,
                                                            float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}

__global__ void fixed_point_quantize_kernel_mask_nearest(float* __restrict__ a,
                                                         float* o, uint8_t* m,
                                                         int size, int sigma,
                                                         float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}


// __global__ void fixed_point_quantize_partial_kernel(float* __restrict__ a,
//                                                     float* o, int size,
//                                                     int sigma, bool use_clamp,
//                                                     float t_min, float t_max, float threshold){
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < size) {
//     o[index] = nearest_round(a[index], sigma);
//     if (use_clamp) {
//       o[index] = clamp_helper(o[index], t_min, t_max);
//     }
//   }
// }

__global__ void fixed_point_quantize_partial_kernel(float* __restrict__ a,
                                                    float* o, int size,
                                                    int sigma, bool use_clamp,
                                                    float t_min, float t_max, float threshold){


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {

        float a_val = a[index];
        float quantized_val = nearest_round(a[index], sigma);

        if (fabsf(a_val - quantized_val) <= threshold) {
            o[index] = quantized_val;
            if (use_clamp) {
              o[index] = clamp_helper(o[index], t_min, t_max);
            }
        } else {
            o[index] = a_val;
        }
    }
}