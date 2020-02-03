#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32
#define ROUND_OFF 50000

#include <stdio.h>
#include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;
typedef Eigen::GpuDevice GPUDevice;

__global__ void PadData(
    const float *in,
    int in_widthheight,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    int channels,
    int padding,
    float *out)
{
    int xy = blockIdx.x * blockDim.x + threadIdx.x;

    int x = xy % in_width;
    int y = xy / in_width;
    int ch = blockIdx.y;
    int n = blockIdx.z;

    if (xy >= in_widthheight)
    {
        out[((n * out_height + y) * out_width + x) * channels + ch] = 0.0;
        return;
    }

    float value = in[((n * in_height + y) * in_width + x) * channels + ch];

    __syncthreads();

    int xpad = x + padding;
    int ypad = y;

    out[((n * out_height + ypad) * out_width + xpad) * channels + ch] = value;
}

void Pad(const GPUDevice &device,
         const float *input,
         int batch_size,
         int input_height,
         int input_width,
         int input_channels,
         int output_height,
         int output_width,
         float *output)
{
    int in_widthheight = input_width * input_height;
    int threads_per_block = 16;
    dim3 totalBlocks((in_widthheight - 1) / threads_per_block + 1, input_channels, batch_size);

    cudaMemset(output, 0, batch_size * output_height * output_width * input_channels * sizeof(float));

    int padding = (output_width - input_width) / 2;

    // LAUNCH KERNEL
    PadData<<<totalBlocks, threads_per_block, 0, device.stream()>>>(
        input,
        in_widthheight,
        input_width,
        input_height,
        output_width,
        output_height,
        input_channels,
        padding,
        output);
}

__global__ void CorrelateData1d(int batch_size,
                                int out_width,
                                int out_height,
                                int out_channels,
                                int out_count,
                                int max_displacement,
                                int x_shift,
                                int neighborhood_grid_width,
                                int kernel_radius,
                                int kernel_size,
                                int stride_1,
                                int stride_2,
                                int in_width_padded,
                                int in_height_padded,
                                int in_channels,
                                const float *input_a,
                                const float *input_b,
                                float *output)
{
    extern __shared__ char patch_data_char[];

    float *patch_data = (float *)patch_data_char;

    // First (upper left) position of kernel upper-left corner in current center
    // position of neighborhood in image 1
    int x1 = blockIdx.x * stride_1 + max_displacement;
    int y1 = blockIdx.y * stride_1;
    int item = blockIdx.z;
    int ch_off = threadIdx.x;

    // Load 3D patch into shared shared memory
    // HEIGHT
    for (int j = 0; j < kernel_size; j++)
    {
        // WIDTH
        for (int i = 0; i < kernel_size; i++)
        {
            int ji_off = ((j * kernel_size) + i) * in_channels;

            // CHANNELS
            for (int ch = ch_off; ch < in_channels; ch += (WARPS_PER_BLOCK * THREADS_PER_WARP))
            {
                int idx1 = ((item * in_height_padded + y1 + j) * in_width_padded + x1 + i) *
                               in_channels +
                           ch;
                int idxPatchData = ji_off + ch;
                patch_data[idxPatchData] = input_a[idx1];
            }
        }
    }

    __syncthreads();

    __shared__ float sum[WARPS_PER_BLOCK * THREADS_PER_WARP];

    // Compute correlation
    for (int out_channel = 0; out_channel < out_channels; out_channel++)
    {
        sum[ch_off] = 0;

        int s2o = (out_channel % neighborhood_grid_width + x_shift) * stride_2;
        int x2 = x1 + s2o;

        // HEIGHT
        for (int j = 0; j < kernel_size; j++)
        {
            // WIDTH
            for (int i = 0; i < kernel_size; i++)
            {
                int ji_off = ((j * kernel_size) + i) * in_channels;

                // CHANNELS
                for (int ch = ch_off; ch < in_channels; ch += (WARPS_PER_BLOCK * THREADS_PER_WARP))
                {
                    int idxPatchData = ji_off + ch;
                    int idx2 = ((item * in_height_padded + y1 + j) * in_width_padded + x2 + i) *
                                   in_channels + ch;

                    sum[ch_off] += patch_data[idxPatchData] * input_b[idx2];
                }
            }
        }

        __syncthreads();

        if (ch_off == 0)
        {
            float total_sum = 0;

            for (int idx = 0; idx < WARPS_PER_BLOCK * THREADS_PER_WARP; idx++)
            {
                total_sum += sum[idx];
            }
            const int sumelems = kernel_size * kernel_size * in_channels;
            const int index = (blockIdx.y * out_width + blockIdx.x) * out_channels + out_channel;

            /* from Caffe:   const int index    = ((out_channel * out_height +
         blockIdx.y) * out_width) + blockIdx.x; */
            output[index + item * out_count] = total_sum / (float)sumelems;

            // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
            // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)
            // n = 0
            // caffe: ((k * H + h) * W + w)  +   n * K * H * W
            // tf: (h * W + w) * K + k       +   n * H * W * K
        }
    }
}

void Correlation1d(const GPUDevice &device,
                   const float *input_a,
                   const float *input_b,
                   const int batch_size,
                   const int out_height,
                   const int out_width,
                   const int out_channels,
                   const int out_count,
                   const int in_height_padded,
                   const int in_width_padded,
                   const int in_channels,
                   int max_displacement,
                   int x_shift,
                   int neighborhood_grid_width,
                   int kernel_radius,
                   int kernel_size,
                   int stride_1,
                   int stride_2,
                   float *output)
{
    dim3 totalBlocksCorr(out_width, out_height, batch_size);
    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);
    const int shared_memory_per_block = (kernel_size * kernel_size) * in_channels;

    CorrelateData1d<<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(float),
                      device.stream()>>>(
        batch_size, out_width, out_height, out_channels, out_count,
        max_displacement, x_shift, neighborhood_grid_width, kernel_radius,
        kernel_size, stride_1, stride_2, in_width_padded, in_height_padded, in_channels,
        input_a, input_b, output);
}

__global__ void CorrelateDataBackward0_1d(const int nthreads,
                                          int item,
                                          int out_width,
                                          int out_height,
                                          int out_channels,
                                          int max_displacement,
                                          int x_shift,
                                          int neighborhood_grid_width,
                                          int kernel_radius,
                                          int stride_1,
                                          int stride_2,
                                          int in_width,
                                          int in_height,
                                          int padded_in_width,
                                          int padded_in_height,
                                          int in_channels,
                                          int in_count_per_sample,
                                          int pad_size,
                                          float *output_a_gradient,
                                          const float *input_b,
                                          const float *gradient)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int k = index % in_channels;                          // channels
        int x = (index / in_channels) % in_width + pad_size;  // w-pos
        int y = (index / in_channels / in_width) % in_height; // h-pos

        // Get X,Y ranges and clamp
        // round_off is a trick to enable integer division with ceil, even for
        // negative numbers
        // We use a large offset, for the inner part not to become negative.
        const int round_off = ROUND_OFF;
        const int round_off_s1 = stride_1 * round_off;

        // We add round_off before_s1 the int division and subtract round_off after
        // it, to ensure the formula matches ceil behavior:
        int xmin = (x - 2 * kernel_radius - max_displacement + round_off_s1 - 1) / stride_1 + 1 -
                   round_off;
        int ymin = (y - 2 * kernel_radius - 0 + round_off_s1 - 1) / stride_1 + 1 -
                   round_off;

        // Same here:
        int xmax = (x - max_displacement + round_off_s1) / stride_1 - round_off;
        int ymax = (y - 0 + round_off_s1) / stride_1 - round_off;

        float sum = 0;

        if ((xmax >= 0) && (ymax >= 0) && (xmin <= out_width - 1) && (ymin <= out_height - 1))
        {
            xmin = max(0, xmin);
            xmax = min(out_width - 1, xmax);

            ymin = max(0, ymin);
            ymax = min(out_height - 1, ymax);

            for (int o = x_shift; o < x_shift + neighborhood_grid_width; o++)
            {
                // Get input_b data:
                int s2o = stride_2 * o;
                int idx_input_b = ((item * padded_in_height + y) * padded_in_width + (x + s2o)) *
                                      in_channels + k;
                float input_b_tmp = input_b[idx_input_b]; // input_b[x+s2o,y,k]

                // Index offset for gradient in following loops:
                int op = (o - x_shift); // index [o,p]

                for (int y = ymin; y <= ymax; y++)
                {
                    for (int x = xmin; x <= xmax; x++)
                    {
                        // gradient[x,y,o,p]
                        int idx_gradient = ((item * out_height + y) * out_width + x) * out_channels + op;
                        sum += gradient[idx_gradient] * input_b_tmp;
                    }
                }
            }
        }
        const int sumelems = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * in_channels;
        const int input_a_idx = (y * in_width + (x - pad_size)) * in_channels + k;
        output_a_gradient[input_a_idx + item * in_count_per_sample] = sum / (float)sumelems;
    }
}

__global__ void CorrelateDataBackward1_1d(const int nthreads,
                                          int item,
                                          int out_width,
                                          int out_height,
                                          int out_channels,
                                          int max_displacement,
                                          int x_shift,
                                          int neighborhood_grid_width,
                                          int kernel_radius,
                                          int stride_1,
                                          int stride_2,
                                          int in_width,
                                          int in_height,
                                          int padded_in_width,
                                          int padded_in_height,
                                          int in_channels,
                                          int in_count_per_sample,
                                          int pad_size,
                                          float *output_b_gradient,
                                          const float *input_a,
                                          const float *gradient)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int k = index % in_channels;                                     // channels
        int x = (index / in_channels) % in_width + pad_size;             // w-pos
        int y = (index / in_channels / in_width) % in_height;            // h-pos

        // round_off is a trick to enable integer division with ceil, even for
        // negative numbers
        // We use a large offset, for the inner part not to become negative.
        const int round_off = ROUND_OFF;
        const int round_off_s1 = stride_1 * round_off;

        float sum = 0;

       // Width (x)
        for (int o = x_shift; o < x_shift + neighborhood_grid_width; o++)
        {
            int s2o = stride_2 * o;
            // Get X,Y ranges and clamp
            // We add round_off before_s1 the int division and subtract round_off
            // after it, to ensure the formula matches ceil behavior:
            int xmin = (x - 2 * kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride_1 +
                        1 - round_off;
            int ymin = (y - 2 * kernel_radius - 0 - 0 + round_off_s1 - 1) / stride_1 +
                        1 - round_off;

            // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
            // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)

            // Same here:
            int xmax = (x - max_displacement - s2o + round_off_s1) / stride_1 - round_off;
            int ymax = (y - 0 - 0 + round_off_s1) / stride_1 - round_off;

            if ((xmax >= 0) && (ymax >= 0) && (xmin <= out_width - 1) && (ymin <= out_height - 1))
            {
                xmin = max(0, xmin);
                xmax = min(out_width - 1, xmax);

                ymin = max(0, ymin);
                ymax = min(out_height - 1, ymax);

                // Get input_a data:
                int idx_input_a = ((item * padded_in_height + y) * padded_in_width + (x - s2o)) *
                                        in_channels + k;
                float input_a_tmp = input_a[idx_input_a];

                // Index offset for gradient in following loops:
                int op = (o - x_shift); // index [o,p]

                for (int y = ymin; y <= ymax; y++)
                {
                    for (int x = xmin; x <= xmax; x++)
                    {
                        int idx_gradient = ((item * out_height + y) * out_width + x) * out_channels + op;
                        sum += gradient[idx_gradient] * input_a_tmp;
                    }
                }
            }
        }
        const int sumelems = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * in_channels;
        const int input_b_idx = (y * in_width + (x - pad_size)) * in_channels + k;
        output_b_gradient[input_b_idx + item * in_count_per_sample] = sum / (float)sumelems;
    }
}

void Correlation1dGradA(const GPUDevice &device,
                        const int batch_size,
                        const int out_width,
                        const int out_height,
                        const int out_channels,
                        const int max_displacement,
                        const int x_shift,
                        const int neighborhood_grid_width,
                        const int kernel_radius,
                        const int stride_1,
                        const int stride_2,
                        const int in_width,
                        const int in_height,
                        const int padded_in_width,
                        const int padded_in_height,
                        const int in_channels,
                        const int in_count_per_sample, // h * w * ch
                        const int pad,
                        const float *input_b,
                        const float *gradient,
                        float *output_a_gradient)
{
    CudaLaunchConfig config = GetCudaLaunchConfig(in_count_per_sample, device);

    for (int n = 0; n < batch_size; n++)
    {
        CorrelateDataBackward0_1d<<<config.block_count, config.thread_per_block, 0,
                                    device.stream()>>>(
            in_count_per_sample,
            n, out_width, out_height, out_channels,
            max_displacement, x_shift, neighborhood_grid_width, kernel_radius,
            stride_1, stride_2,
            in_width, in_height, padded_in_width, padded_in_height, in_channels, in_count_per_sample, pad,
            output_a_gradient, input_b, gradient);
    }
}

void Correlation1dGradB(const GPUDevice &device,
                        const int batch_size,
                        const int out_width,
                        const int out_height,
                        const int out_channels,
                        const int max_displacement,
                        const int x_shift,
                        const int neighborhood_grid_width,
                        const int kernel_radius,
                        const int stride_1,
                        const int stride_2,
                        const int in_width,
                        const int in_height,
                        const int padded_in_width,
                        const int padded_in_height,
                        const int in_channels,
                        const int in_count_per_sample,
                        const int pad,
                        const float *input_a,
                        const float *gradient,
                        float *output_b_gradient)
{
    CudaLaunchConfig config = GetCudaLaunchConfig(in_count_per_sample, device);

    for (int n = 0; n < batch_size; n++)
    {
        CorrelateDataBackward1_1d<<<config.block_count, config.thread_per_block, 0,
                                    device.stream()>>>(
            in_count_per_sample,
            n, out_width, out_height, out_channels,
            max_displacement, x_shift, neighborhood_grid_width, kernel_radius,
            stride_1, stride_2,
            in_width, in_height, padded_in_width, padded_in_height, in_channels, in_count_per_sample, pad,
            output_b_gradient, input_a, gradient);
    }
}

#endif // GOOGLE_CUDA
